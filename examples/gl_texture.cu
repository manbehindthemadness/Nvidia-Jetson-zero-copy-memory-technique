#include <GL/gl.h>
#include <GL/glx.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

constexpr int side = 32;

__global__ void write_pixels(cudaSurfaceObject_t surface) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    surf2Dwrite(make_uchar4(x, y, 67, 255), surface, x * sizeof(uchar4), y);
}

struct Texture {
    Display *display = nullptr;
    GLXContext context = nullptr;
    GLXPbuffer drawable = 0;
    GLuint name = 0;
    cudaGraphicsResource_t resource = nullptr;
    cudaSurfaceObject_t surface = 0;
    bool mapped = false;

    ~Texture() {
        if (mapped) cudaStreamSynchronize(nullptr);
        if (surface) cudaDestroySurfaceObject(surface);
        if (mapped) cudaGraphicsUnmapResources(1, &resource);
        if (resource) cudaGraphicsUnregisterResource(resource);
        if (name) glDeleteTextures(1, &name);
        if (context) {
            glXMakeContextCurrent(display, None, None, nullptr);
            glXDestroyContext(display, context);
        }
        if (drawable) glXDestroyPbuffer(display, drawable);
        if (display) XCloseDisplay(display);
    }
};

int main() try {
    Texture texture;
    texture.display = XOpenDisplay(nullptr);
    if (!texture.display)
        throw std::runtime_error("Could not open the X11 display");
    int config[] = {GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT, GLX_RENDER_TYPE, GLX_RGBA_BIT, None};
    int count = 0;
    GLXFBConfig *configs = glXChooseFBConfig(texture.display, DefaultScreen(texture.display), config, &count);
    if (!count)
        throw std::runtime_error("Could not choose a GLX framebuffer config");
    int size[] = {GLX_PBUFFER_WIDTH, side, GLX_PBUFFER_HEIGHT, side, None};
    texture.drawable = glXCreatePbuffer(texture.display, configs[0], size);
    texture.context = glXCreateNewContext(texture.display, configs[0], GLX_RGBA_TYPE, nullptr, True);
    XFree(configs);
    if (!texture.context || !texture.drawable ||
        !glXMakeContextCurrent(texture.display, texture.drawable, texture.drawable, texture.context))
        throw std::runtime_error("Could not activate the GLX context");

    glGenTextures(1, &texture.name);
    glBindTexture(GL_TEXTURE_2D, texture.name);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, side, side, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    if (glGetError() != GL_NO_ERROR)
        throw std::runtime_error("Could not allocate the OpenGL texture");
    check(cudaGraphicsGLRegisterImage(&texture.resource, texture.name, GL_TEXTURE_2D,
                                      cudaGraphicsRegisterFlagsSurfaceLoadStore), "Register GL texture");
    check(cudaGraphicsMapResources(1, &texture.resource), "Map GL texture");
    texture.mapped = true;
    cudaArray_t array;
    check(cudaGraphicsSubResourceGetMappedArray(&array, texture.resource, 0, 0), "Get mapped CUDA array");
    cudaResourceDesc description{};
    description.resType = cudaResourceTypeArray;
    description.res.array.array = array;
    check(cudaCreateSurfaceObject(&texture.surface, &description), "Create mapped surface");
    write_pixels<<<dim3(4, 4), dim3(8, 8)>>>(texture.surface);
    check(cudaGetLastError(), "Launch write_pixels");
    check(cudaStreamSynchronize(nullptr), "Wait before destroying the surface");
    check(cudaDestroySurfaceObject(texture.surface), "Destroy surface");
    texture.surface = 0;
    check(cudaGraphicsUnmapResources(1, &texture.resource), "Unmap GL texture");
    texture.mapped = false;

    // Readback is only for this validation. Rendering can sample the texture directly.
    std::array<unsigned char, 4 * side * side> pixels{};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    if (glGetError() != GL_NO_ERROR)
        throw std::runtime_error("Could not read the GL texture for validation");
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            int offset = 4 * (y * side + x);
            if (pixels[offset] != x || pixels[offset + 1] != y ||
                pixels[offset + 2] != 67 || pixels[offset + 3] != 255)
                throw std::runtime_error("OpenGL did not observe the expected CUDA pixel");
        }
    }
    std::cout << "CUDA wrote and OpenGL verified " << side * side << " pixels in the same texture\n";
    return EXIT_SUCCESS;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
}
