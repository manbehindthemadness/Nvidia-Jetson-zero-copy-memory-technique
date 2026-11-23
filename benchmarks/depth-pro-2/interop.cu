#include <GL/gl.h>
#include <GL/glx.h>
#include <X11/keysym.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CUDA(call) check((call), #call)

void check_gl(const char *operation) {
    GLenum status = glGetError();
    if (status != GL_NO_ERROR)
        throw std::runtime_error(std::string(operation) + ": OpenGL error " + std::to_string(status));
}

template <typename Function>
int call(Function function) {
    try {
        function();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}

template <typename Value>
void allocate(Value **pointer, size_t count) {
    CUDA(cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(Value)));
}

constexpr int sample_step = 8;
constexpr int threads = 256;

__device__ float finite_depth(float value) {
    return isfinite(value) ? value : 0.0f;
}

__global__ void sample_depth(const float *depth, float *samples, int width, int columns, int count) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < count) {
        int row = index / columns * sample_step;
        int column = index % columns * sample_step;
        samples[index] = finite_depth(depth[row * width + column]);
    }
}

__global__ void smooth_scale(const float *sorted, int count, float *scale, bool initialize) {
    double rank = .98 * (count - 1);
    int lower = static_cast<int>(rank);
    int upper = min(lower + 1, count - 1);
    double limit = fmax(.01, sorted[lower] + (rank - lower) * (sorted[upper] - sorted[lower]));
    *scale = static_cast<float>(initialize ? limit : .92 * *scale + .08 * limit);
}

template <bool Direct>
__global__ void compose(const float *rgb, const float *depth, const float *scale,
                        const uchar4 *lut, int width, int height,
                        uchar4 *staging, cudaSurfaceObject_t surface) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= 2 * width || y >= height)
        return;
    int pixel = y * width + x % width;
    uchar4 color;
    if (x < width) {
        int plane = width * height;
        color = make_uchar4(__float2uint_rn(__saturatef(rgb[pixel]) * 255),
                            __float2uint_rn(__saturatef(rgb[pixel + plane]) * 255),
                            __float2uint_rn(__saturatef(rgb[pixel + 2 * plane]) * 255), 255);
    } else {
        int luma = static_cast<int>(fminf(255, fmaxf(0, finite_depth(depth[pixel]) * (255 / *scale))));
        color = lut[luma];
    }
    if constexpr (Direct)
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    else
        staging[y * 2 * width + x] = color;
}

class Viewer {
public:
    Viewer(int width, int height) : width(width), height(height),
        sample_columns((width + sample_step - 1) / sample_step),
        sample_count(sample_columns * ((height + sample_step - 1) / sample_step)) {}

    void open(const unsigned char *colors) {
        display = XOpenDisplay(nullptr);
        if (!display)
            throw std::runtime_error("Could not open X11 display");
        int attributes[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
                            GLX_BLUE_SIZE, 8, None};
        std::unique_ptr<XVisualInfo, decltype(&XFree)> visual(
            glXChooseVisual(display, DefaultScreen(display), attributes), XFree);
        if (!visual)
            throw std::runtime_error("Could not choose a double-buffered GLX visual");
        context = glXCreateContext(display, visual.get(), nullptr, True);
        if (!context)
            throw std::runtime_error("Could not create a GLX context");
        Window root = RootWindow(display, visual->screen);
        XSetWindowAttributes settings{};
        colormap = XCreateColormap(display, root, visual->visual, AllocNone);
        settings.colormap = colormap;
        settings.event_mask = StructureNotifyMask | KeyPressMask;
        viewport_width = std::min(1280, 2 * width);
        viewport_height = std::max(1, viewport_width * height / (2 * width));
        window = XCreateWindow(display, root, 0, 0, viewport_width, viewport_height, 0,
                               visual->depth, InputOutput, visual->visual,
                               CWColormap | CWEventMask, &settings);
        wm_delete = XInternAtom(display, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display, window, &wm_delete, 1);
        XStoreName(display, window, "Depth Pro 2 CUDA / OpenGL experiment");
        XMapWindow(display, window);
        if (!glXMakeCurrent(display, window, context))
            throw std::runtime_error("Could not activate the GLX context");
        renderer = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
        auto swap_interval = reinterpret_cast<PFNGLXSWAPINTERVALEXTPROC>(
            glXGetProcAddressARB(reinterpret_cast<const GLubyte *>("glXSwapIntervalEXT")));
        const char *extensions = glXQueryExtensionsString(display, visual->screen);
        if (swap_interval && extensions && std::string(extensions).find("GLX_EXT_swap_control") != std::string::npos)
            swap_interval(display, window, 0);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2 * width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glEnable(GL_TEXTURE_2D);
        check_gl("Create display texture");
        CUDA(cudaGraphicsGLRegisterImage(&texture_resource, texture, GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsSurfaceLoadStore));
        allocate(&sort_input, sample_count);
        allocate(&sort_output, sample_count);
        allocate(&scale, 1);
        allocate(&lut, 256);
        allocate(&staging, 2 * width * height);
        CUDA(cudaMemcpy(lut, colors, 256 * sizeof(uchar4), cudaMemcpyHostToDevice));
        CUDA(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, sort_input, sort_output, sample_count));
        CUDA(cudaMalloc(&sort_scratch, sort_bytes));
    }

    ~Viewer() {
        if (!display)
            return;
        if (texture_resource)
            cudaGraphicsUnregisterResource(texture_resource);
        if (sort_input) cudaFree(sort_input);
        if (sort_output) cudaFree(sort_output);
        if (scale) cudaFree(scale);
        if (lut) cudaFree(lut);
        if (staging) cudaFree(staging);
        if (sort_scratch) cudaFree(sort_scratch);
        if (texture)
            glDeleteTextures(1, &texture);
        if (context) {
            glXMakeCurrent(display, None, nullptr);
            glXDestroyContext(display, context);
        }
        if (window) XDestroyWindow(display, window);
        if (colormap) XFreeColormap(display, colormap);
        XCloseDisplay(display);
    }

    bool events() {
        while (XPending(display)) {
            XEvent event;
            XNextEvent(display, &event);
            if (event.type == ConfigureNotify) {
                viewport_width = event.xconfigure.width;
                viewport_height = event.xconfigure.height;
            } else if (event.type == ClientMessage && static_cast<Atom>(event.xclient.data.l[0]) == wm_delete) {
                return false;
            } else if (event.type == KeyPress) {
                auto key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_Escape || key == XK_q)
                    return false;
            }
        }
        return true;
    }

    void upload(const unsigned char *rgba) {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2 * width, height, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    }

    void gpu(const float *rgb, const float *depth, cudaStream_t stream, bool direct) {
        sample_depth<<<(sample_count + threads - 1) / threads, threads, 0, stream>>>(
            depth, sort_input, width, sample_columns, sample_count);
        CUDA(cudaGetLastError());
        CUDA(cub::DeviceRadixSort::SortKeys(sort_scratch, sort_bytes, sort_input, sort_output,
                                           sample_count, 0, 32, stream));
        smooth_scale<<<1, 1, 0, stream>>>(sort_output, sample_count, scale, initialize_scale);
        CUDA(cudaGetLastError());
        initialize_scale = false;

        const dim3 block(32, 8);
        const dim3 grid((2 * width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        if (!direct) {
            compose<false><<<grid, block, 0, stream>>>(rgb, depth, scale, lut, width, height, staging, 0);
            CUDA(cudaGetLastError());
        }

        CUDA(cudaGraphicsMapResources(1, &texture_resource, stream));
        cudaSurfaceObject_t surface = 0;
        try {
            cudaArray_t array;
            CUDA(cudaGraphicsSubResourceGetMappedArray(&array, texture_resource, 0, 0));
            if (direct) {
                cudaResourceDesc description{};
                description.resType = cudaResourceTypeArray;
                description.res.array.array = array;
                CUDA(cudaCreateSurfaceObject(&surface, &description));
                compose<true><<<grid, block, 0, stream>>>(rgb, depth, scale, lut, width, height, nullptr, surface);
                CUDA(cudaGetLastError());
            } else {
                size_t stride = 2 * width * sizeof(uchar4);
                CUDA(cudaMemcpy2DToArrayAsync(array, 0, 0, staging, stride, stride, height,
                                              cudaMemcpyDeviceToDevice, stream));
            }
            // Keep the mapped array and the per-map surface alive through the kernel.
            CUDA(cudaStreamSynchronize(stream));
            if (surface) {
                CUDA(cudaDestroySurfaceObject(surface));
                surface = 0;
            }
        } catch (...) {
            cudaStreamSynchronize(stream);
            if (surface) cudaDestroySurfaceObject(surface);
            cudaGraphicsUnmapResources(1, &texture_resource, stream);
            throw;
        }
        CUDA(cudaGraphicsUnmapResources(1, &texture_resource, stream));
    }

    void draw() {
        glViewport(0, 0, viewport_width, viewport_height);
        glBegin(GL_TRIANGLE_STRIP);
        glTexCoord2f(0, 1); glVertex2f(-1, -1);
        glTexCoord2f(1, 1); glVertex2f( 1, -1);
        glTexCoord2f(0, 0); glVertex2f(-1,  1);
        glTexCoord2f(1, 0); glVertex2f( 1,  1);
        glEnd();
        glFinish();
        glXSwapBuffers(display, window);
    }

    void read(unsigned char *rgba) {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba);
        check_gl("Read display texture for validation");
    }

    bool initialize_scale = true;
    std::string renderer;

private:
    int width, height, sample_columns, sample_count;
    int viewport_width = 0, viewport_height = 0;
    Display *display = nullptr;
    Window window = 0;
    Colormap colormap = 0;
    Atom wm_delete = 0;
    GLXContext context = nullptr;
    GLuint texture = 0;
    cudaGraphicsResource_t texture_resource = nullptr;
    float *sort_input = nullptr, *sort_output = nullptr, *scale = nullptr;
    uchar4 *lut = nullptr, *staging = nullptr;
    void *sort_scratch = nullptr;
    size_t sort_bytes = 0;
};
} // namespace

extern "C" {
const char *viewer_error() { return last_error.c_str(); }

int viewer_create(int width, int height, const unsigned char *palette, void **result) {
    return call([&] {
        auto viewer = std::make_unique<Viewer>(width, height);
        viewer->open(palette);
        *result = viewer.release();
    });
}

const char *viewer_renderer(void *viewer) { return static_cast<Viewer *>(viewer)->renderer.c_str(); }
void viewer_reset(void *viewer) { static_cast<Viewer *>(viewer)->initialize_scale = true; }
int viewer_events(void *viewer) { return static_cast<Viewer *>(viewer)->events(); }
void viewer_upload(void *viewer, const unsigned char *rgba) { static_cast<Viewer *>(viewer)->upload(rgba); }
int viewer_gpu(void *viewer, const float *rgb, const float *depth, cudaStream_t stream, int direct) {
    return call([&] { static_cast<Viewer *>(viewer)->gpu(rgb, depth, stream, direct != 0); });
}
void viewer_draw(void *viewer) { static_cast<Viewer *>(viewer)->draw(); }
int viewer_read(void *viewer, unsigned char *rgba) { return call([&] { static_cast<Viewer *>(viewer)->read(rgba); }); }
void viewer_destroy(void *viewer) { delete static_cast<Viewer *>(viewer); }
}
