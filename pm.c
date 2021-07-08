/*
m  &&  t ./mviz

t nvcc  mviz-cov2-s.cu -cubin -o mviz-cov2-s.cubin  -cudart none -arch=sm_70 -use_fast_math -O3
t gcc-8  mviz.c -o viz  -lcuda -lX11 -lX11-xcb -lxcb -lGL  -Ofast -flto -fwhole-program  &&  t ./mviz

1920x1080 titan_v 0% (fill rect): 350 us

# CUDA Driver API!

The use of multiple CUcontexts per device within a single process will SUBSTANTIALLY DEGRADE PERFORMANCE and is strongly discouraged!
Instead, it's highly recommended that the implicit one-to-one device-to-context mapping for the process provided by the CUDA Runtime API be used.
If a non-primary CUcontext created by the CUDA Driver API is current to a thread then the CUDA Runtime API calls to that thread will operate on that CUcontext, with some exceptions.

It's recommended that the primary context not be deinitialized except just before exit or to recover from an unspecified launch failure.
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <locale.h>
#include "pm.h"

#define m_checksys(FN_ST, FN_NAME)     if(((i64)(FN_ST))==-1) printf("  \x1b[92mCTX \x1b[32m%s\x1b[91m:\x1b[0mL\x1b[94m%d\x1b[91m:\x1b[35m%s\x1b[0m  \x1b[92mFN \x1b[35m%s\x1b[0m  \x1b[92mST \x1b[31m%d\x1b[91m:\x1b[37m%s\x1b[0m!  \x1b[92mARG \x1b[91m[\x1b[37m%s\x1b[91m;\x1b[91m]\x1b[0m\n", __FILE__,__LINE__,__func__, (FN_NAME), errno,strerror(errno), "???");
#define m_max(A, B)                    ({  __auto_type a=(A);  __auto_type b=(B);  a<b ? b : a;  })
#define m_min(A, B)                    ({  __auto_type a=(A);  __auto_type b=(B);  a<b ? a : b;  })
#define m_divceilu(DIVIDEND, DIVISOR)  (((DIVIDEND)%(DIVISOR)) ? (DIVIDEND)/(DIVISOR)+1 : (DIVIDEND)/(DIVISOR))  // BEWARE! Expensive, since there's a mod!




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block  libs & stuff! */

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  CUDA Driver API graphics interop! */
#include <GL/glx.h>  // GL_GLEXT_PROTOTYPES GLX_GLXEXT_PROTOTYPES
#include <X11/Xlib-xcb.h>  // includes X11/Xlib.h and xcb/xcb.h!
#include <cudaGL.h>

#define cu_check(CU__ST){  CUresult cu__st = (CU__ST);  if(cu__st != CUDA_SUCCESS){  const char* txt;  printf("\x1b[91mFAIL  \x1b[32m%s\x1b[33m:\x1b[0mL\x1b[94m%d\x1b[33m:\x1b[35m%s  \x1b[32mCUDA\x1b[0m  \x1b[31m%x\x1b[0m\x1b[91m:\x1b[0m", __FILE__,__LINE__,__func__,cu__st);  cuGetErrorName(cu__st,&txt); printf("\x1b[35m%s\x1b[0m  ",txt);  cuGetErrorString(cu__st,&txt); printf("\x1b[37m%s\x1b[0m  ",txt);  putchar(0x0a);  }  }
#define gl_check(){  GLenum err;  while((err=glGetError()) != GL_NO_ERROR){ printf("\n\x1b[91mGL_ERROR  \x1b[32m%s\x1b[0m:\x1b[94mL%d\x1b[0m  \x1b[33m%s\x1b[0m  0x\x1b[32m%04x\x1b[0m:\x1b[35m%d\x1b[0m\n", __FILE__,__LINE__,__func__, err,err); }  }

GLAPI void   APIENTRY glGenVertexArrays(       GLsizei n, GLuint *arrays);
GLAPI void   APIENTRY glBindVertexArray(       GLuint array);
GLAPI void   APIENTRY glDeleteVertexArrays(    GLsizei n, const GLuint *arrays);
GLAPI void   APIENTRY glGenBuffers(            GLsizei n, GLuint *buffers);
GLAPI void   APIENTRY glBindBuffer(            GLenum target, GLuint buffer);
GLAPI void   APIENTRY glDeleteBuffers(         GLsizei n, const GLuint *buffers);
GLAPI void   APIENTRY glBufferStorage(         GLenum target, GLsizeiptr size, const void *data, GLbitfield flags);
GLAPI void   APIENTRY glGenTextures(           GLsizei n, GLuint *textures);
GLAPI void   APIENTRY glBindTexture(           GLenum target, GLuint texture);
GLAPI void   APIENTRY glDeleteTextures(        GLsizei n, const GLuint *textures);
GLAPI void   APIENTRY glTexParameteri(         GLenum target, GLenum pname, GLint param );
GLAPI void   APIENTRY glTexStorage2D(          GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
GLAPI void   APIENTRY glGenFramebuffers(       GLsizei n, GLuint *framebuffers);
GLAPI void   APIENTRY glBindFramebuffer(       GLenum target, GLuint framebuffer);
GLAPI void   APIENTRY glDeleteFramebuffers(    GLsizei n, const GLuint *framebuffers);
GLAPI void   APIENTRY glFramebufferTexture2D(  GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);
GLAPI GLenum APIENTRY glCheckFramebufferStatus(GLenum target);
GLAPI void   APIENTRY glViewport(              GLint x, GLint y, GLsizei width, GLsizei height);
GLAPI void   APIENTRY glBlitFramebuffer(       GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter);

GLXContext glXCreateContextAttribsARB(Display* dpy, GLXFBConfig config, GLXContext share_context, Bool direct, const int* attrib_list);

GLXFBConfig glx_fbconfig_get(Display* x11_display, int x11_screen, int* glx_fbconfig_attrs){  // Grab the first fbconfig out of a list of matching fbconfigs!
  int glx_nfbconfigs;
  GLXFBConfig* glx_fbconfigs = glXChooseFBConfig(x11_display, x11_screen, glx_fbconfig_attrs, &glx_nfbconfigs);  if(glx_nfbconfigs==0){ printf("\x1b[32m%s\x1b[0m\x1b[91m:\x1b[0mL\x1b[94m%d\x1b[0m\x1b[91m:\x1b[0m\x1b[35m%s\x1b[0m  \x1b[33m%s\x1b[0m  %s\n", __FILE__,__LINE__,__func__, "glXChooseFBConfig", "No GLX framebuffer configuration (GLXFBConfig) matches the requested attributes!"); return 0; }
  GLXFBConfig  glx_fbconfig  = glx_fbconfigs[0];  // for(int i=1s; i < glx_nfbconfigs; ++i)  XFree(glx_fbconfigs[i]);  // WARN! Do we need to free every element owned by `GLXFBConfig*` or just `GLXFBConfig*` itself?
  XFree(glx_fbconfigs);
  return glx_fbconfig;
}
xcb_screen_t* xcb_screen_get(xcb_connection_t* xcb_connection, int screen_num){
  const xcb_setup_t* setup = xcb_get_setup(xcb_connection);
  for(xcb_screen_iterator_t it = xcb_setup_roots_iterator(setup);  it.rem;  --screen_num, xcb_screen_next(&it))
    if(screen_num==0) return it.data;
  return NULL;
}

typedef struct{  // CUDA-OpenGL window!
  Display*           x11_display;  // The GLX context involves a glx_context and an xcb_window!
  GLXContext         glx_context;
  xcb_connection_t*  xcb_connection;  // Major handle!
  xcb_screen_t*      xcb_screen;
  xcb_colormap_t     xcb_colormap;
  xcb_window_t       xcb_window;      // Major handle!
  GLuint             gl_vao,gl_pbo, gl_tex,gl_fbo;
  CUgraphicsResource cu_gres;
  CUcontext          cu_ctx;
  uint16_t           dim_h,dim_w;
  size_t             nbytes;
  u32*               cu_data;
}win_t;

void win_init2(win_t* win, CUcontext cu_ctx){  CUresult st;  // int ngldevs=0;  CUdevice gldevs[0x400];  st=cuGLGetDevices(&ngldevs,gldevs,0x400,CU_GL_DEVICE_LIST_ALL); cu_check(st);  printf("\x1b[32mcuda  \x1b[35mgl  \x1b[0mndevs \x1b[31m%d  \x1b[0m\n", ngldevs);
  win->cu_ctx = cu_ctx;
  st=cuCtxPushCurrent(win->cu_ctx);                                                                     cu_check(st);
  st=cuGraphicsGLRegisterBuffer(&win->cu_gres, win->gl_pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);  cu_check(st);
  st=cuGraphicsMapResources(1, &win->cu_gres, 0);                                                       cu_check(st);
  st=cuGraphicsResourceGetMappedPointer((CUdeviceptr*)&win->cu_data, &win->nbytes, win->cu_gres);       cu_check(st);
  st=cuGraphicsUnmapResources(1, &win->cu_gres, 0);                                                     cu_check(st);
  st=cuCtxPopCurrent(NULL);                                                                             cu_check(st);
}

win_t* win_init(uint16_t dim_h, uint16_t dim_w){
  win_t* win  = malloc(sizeof(win_t));
  win->dim_h  = dim_h;
  win->dim_w  = dim_w;

  int glx_fbconfig_args[] = {  // glXChooseFBConfig()
    GLX_SAMPLES,      0,
    GLX_DOUBLEBUFFER, 0,  // We *never* get a doublebuffered context from GLX (since GLX sucks)! Rather, we implement our own doublebuffering using OpenGL FBO's!
  0};
  int glx_context_args[] = {  // glXCreateContextAttribsARB()
    GLX_CONTEXT_MAJOR_VERSION_ARB,   4,
    GLX_CONTEXT_MINOR_VERSION_ARB,   6,
    GLX_CONTEXT_PROFILE_MASK_ARB,    GLX_CONTEXT_CORE_PROFILE_BIT_ARB,        // GLX_CONTEXT_CORE_PROFILE_BIT_ARB GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB
    GLX_CONTEXT_FLAGS_ARB,           GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,  // GLX_CONTEXT_DEBUG_BIT_ARB        GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB
    GLX_CONTEXT_OPENGL_NO_ERROR_ARB, 1,                                       // 0, 1  // It works!! =D
  0};

  // ----------------------------------------------------------------
  win->x11_display = XOpenDisplay(NULL);  // If @display_name is NULL, it defaults to the value of the DISPLAY environment variable!
  int x11_screen   = DefaultScreen(win->x11_display);
  XSetEventQueueOwner(win->x11_display, XCBOwnsEventQueue);  // Given a shared Xlib/XCB connection, make XCB own the event queue!

  win->xcb_connection = XGetXCBConnection(win->x11_display);  // Instead of calling xcb_connect(), we call XGetXCBConnection()! I don't think we need to (or even *can*) xcb_disconnect() this xcb_connection_t!
  win->xcb_screen     = xcb_screen_get(win->xcb_connection, x11_screen);

  GLXFBConfig glx_fbconfig = glx_fbconfig_get(win->x11_display, x11_screen, glx_fbconfig_args);                      // Last arg can also be NULL!
  win->glx_context         = glXCreateContextAttribsARB(win->x11_display, glx_fbconfig, NULL, 1, glx_context_args);  // The OpenGL CONTEXT is most important! glXCreateNewContext() is for legacy contexts (OpenGL 3 compatibility)! glXCreateContext() is (probably) for even older contexts! glXCreateContextAttribs() is for modern contexts (OpenGL 3 core/compatibility and OpenGL 4 core/compatibility)!
  int glx_visual_id;         glXGetFBConfigAttrib(win->x11_display, glx_fbconfig, GLX_VISUAL_ID, &glx_visual_id);    // Get the GLX_VISUAL_ID of the given glx_fbconfig, so that we can create a compatible xcb_colormap and xcb_window!

  win->xcb_colormap = xcb_generate_id(win->xcb_connection); xcb_create_colormap(win->xcb_connection, XCB_COLORMAP_ALLOC_NONE, win->xcb_colormap, win->xcb_screen->root, glx_visual_id);
  win->xcb_window   = xcb_generate_id(win->xcb_connection); xcb_create_window(win->xcb_connection, win->xcb_screen->root_depth, win->xcb_window, win->xcb_screen->root, 0,0,win->dim_w,win->dim_h, 0,XCB_WINDOW_CLASS_INPUT_OUTPUT, glx_visual_id, XCB_CW_BACK_PIXMAP|XCB_CW_EVENT_MASK|XCB_CW_COLORMAP, (uint32_t[]){XCB_BACK_PIXMAP_NONE, XCB_EVENT_MASK_KEY_PRESS|XCB_EVENT_MASK_KEY_RELEASE|XCB_EVENT_MASK_BUTTON_PRESS|XCB_EVENT_MASK_BUTTON_RELEASE|XCB_EVENT_MASK_POINTER_MOTION|XCB_EVENT_MASK_EXPOSURE|XCB_EVENT_MASK_STRUCTURE_NOTIFY, win->xcb_colormap});
  xcb_map_window(win->xcb_connection, win->xcb_window);  //  Map the window ASAP, and flush right away?
  xcb_flush(win->xcb_connection);
  xcb_configure_window(win->xcb_connection, win->xcb_window, XCB_CONFIG_WINDOW_X|XCB_CONFIG_WINDOW_Y, (uint32_t[]){(win->xcb_screen->width_in_pixels-win->dim_w)/2,(win->xcb_screen->height_in_pixels-win->dim_w)/2});  // The values of this array MUST match the order of the enum where all the masks are defined!

  uint32_t gc_fg   = xcb_generate_id(win->xcb_connection);  xcb_create_gc(win->xcb_connection, gc_fg, win->xcb_screen->root, XCB_GC_FOREGROUND|XCB_GC_GRAPHICS_EXPOSURES, (uint32_t[]){0x0088ff,XCB_EXPOSURES_NOT_ALLOWED});
  uint32_t gc_bg   = xcb_generate_id(win->xcb_connection);  xcb_create_gc(win->xcb_connection, gc_bg, win->xcb_screen->root, XCB_GC_FOREGROUND|XCB_GC_GRAPHICS_EXPOSURES, (uint32_t[]){0x080808,XCB_EXPOSURES_NOT_ALLOWED});
  uint32_t bg_dark = xcb_generate_id(win->xcb_connection);  xcb_create_pixmap(win->xcb_connection, win->xcb_screen->root_depth, bg_dark, win->xcb_screen->root, win->dim_w,win->dim_h);
  xcb_poly_fill_rectangle(win->xcb_connection, bg_dark, gc_bg, 1, (xcb_rectangle_t[]){{0,0, win->dim_w,win->dim_h}});  // Can you draw to "unmapped" pixmaps? You certainly can't draw to unmapped windows! But pixmaps don't even have a `mapped` state, so...
  xcb_copy_area(win->xcb_connection, bg_dark, win->xcb_window, gc_bg, 0,0, 0,0, win->dim_w,win->dim_h);  // CLEAR the window!
  uint8_t poly_txt[2+0xff] = {10,0, 0x4c,0x6f,0x61,0x64,0x69,0x6e,0x67,0x2e,0x2e,0x2e/*Loading...*/};  // Byte 0: length. Byte 1: delta. Bytes [2..257): ascii text. This is the structure that xcb_poly_text_8() accepts for drawing!  https://lists.freedesktop.org/archives/xcb/2011-January/006732.html
  xcb_poly_text_8(win->xcb_connection, win->xcb_window, gc_fg, (win->dim_w-10*5)/2,win->dim_h/2, 2+poly_txt[0], (const uint8_t*)poly_txt);  // Can draw fonts WITHOUT background! Just AWESOME! xcb_poly_text_8_checked() is slower because it BLOCKS (by flushing errors)!
  xcb_free_pixmap(win->xcb_connection, bg_dark);
  xcb_free_gc(    win->xcb_connection, gc_bg);
  xcb_free_gc(    win->xcb_connection, gc_fg);

  // ---------------------------------------------------------------- win->glx_window = glXCreateWindow(win->x11_display, glx_fbconfig, win->xcb_window, NULL);  // Create an on-screen rendering area from an existing X window that was created with a visual matching config. The XID of the GLXWindow is returned. Any GLX rendering context that was created with respect to config can be used to render into this window. Use glXMakeContextCurrent to associate the rendering area with a GLX rendering context! glXCreateWindow() requires an X window to be associated with the GLX window drawable!
  glXMakeContextCurrent(win->x11_display, win->xcb_window,win->xcb_window, win->glx_context);  // Bind a context with a drawable to the CURRENT THREAD! The context/drawable pair becomes the current context and current drawable, and it is used by all OpenGL commands until glXMakeCurrent() is called with different arg! Make OpenGL context CURRENT! THIS is the slow part! Everything until this part is FAST!

  // ----------------------------------------------------------------
  glGenVertexArrays(1, &win->gl_vao);
  glBindVertexArray(win->gl_vao);

  glGenBuffers(1, &win->gl_pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, win->gl_pbo);
  glBufferStorage(GL_PIXEL_UNPACK_BUFFER, 4*win->xcb_screen->height_in_pixels*win->xcb_screen->width_in_pixels, NULL, GL_MAP_WRITE_BIT);

  glGenTextures(1, &win->gl_tex);
  glBindTexture(GL_TEXTURE_RECTANGLE, win->gl_tex);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAX_LEVEL,  0);
  glTexStorage2D(GL_TEXTURE_RECTANGLE, 1, GL_RGBA8, win->xcb_screen->width_in_pixels,win->xcb_screen->height_in_pixels);

  glGenFramebuffers(1, &win->gl_fbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, win->gl_fbo);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, win->gl_tex, 0);  GLenum gl_st=glCheckFramebufferStatus(GL_READ_FRAMEBUFFER); if(gl_st!=GL_FRAMEBUFFER_COMPLETE) printf("\x1b[91mWARN\x1b[0m  \x1b[33m%s  \x1b[0m%d  \x1b[0m\n", "glCheckFramebufferStatus", gl_st);
  glViewport(0,0, win->dim_w,win->dim_h);
  gl_check();
  return win;
}

void win_free(win_t* win){  CUresult st;
  st=cuCtxPushCurrent(win->cu_ctx);               cu_check(st);
  st=cuGraphicsUnregisterResource(win->cu_gres);  cu_check(st);
  st=cuCtxPopCurrent(NULL);                       cu_check(st);

  glDeleteFramebuffers(1, &win->gl_fbo);
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
  glDeleteTextures(1, &win->gl_tex);
  glBindTexture(GL_TEXTURE_RECTANGLE, 0);
  glDeleteBuffers(1, &win->gl_pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glDeleteVertexArrays(1, &win->gl_vao);
  glBindVertexArray(0);  gl_check();

  glXMakeContextCurrent(win->x11_display, None,None, NULL);  // To release the current context without assigning a new one, call w/ draw/read set to `None` and ctx set to `NULL`!
  glXDestroyContext(win->x11_display, win->glx_context);  // If the GLX context is not current to any thread, destroy it immediately. Else, destroy it when it becomes not current to any thread!
  xcb_destroy_window(win->xcb_connection, win->xcb_window);
  xcb_free_colormap(win->xcb_connection, win->xcb_colormap);
  XCloseDisplay(win->x11_display);  // No need to call `xcb_disconnect`, since it causes an error?

  free(win);
}

void win_draw(win_t* win){
  glTexSubImage2D(GL_TEXTURE_RECTANGLE,0, 0,0,win->dim_w,win->dim_h, GL_RGBA,GL_UNSIGNED_BYTE, NULL);
  glBlitFramebuffer(0,0,win->dim_w,win->dim_h, 0,0,win->dim_w,win->dim_h, GL_COLOR_BUFFER_BIT,GL_NEAREST);  // When the color buffer is transferred, values are taken from the read buffer of the specified read framebuffer and written to each of the draw buffers of the specified DRAW_FRAMEBUFFER!
}

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  time */
#include <time.h>
typedef struct{  double t0,t1;  }dt_t;
static __inline int64_t dt_abs(){          struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);  return 1000000000ll*ts.tv_sec + ts.tv_nsec;  }  // We return the epoch in nanoseconds!
static __inline double  dt_del(dt_t* dt){  return (dt->t1 - dt->t0) / 1e9;  }
static __inline void    dt_ini(dt_t* dt){  dt->t0 = dt_abs();  }
static __inline void    dt_end(dt_t* dt){  dt->t1 = dt_abs();  }

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#include <poll.h>
xcb_generic_event_t* xcb_ev_poll(xcb_connection_t* xcb_connection, int timeout_ms){  // `xcb_generic_event_t` is a polymorphic data structure! The first 8 bits tell you how to cast it, and depending on how you cast it, the interpretation of its binary layout (which is fixed in width) changes!
  struct pollfd pfd;
  pfd.events    = POLLIN;  // POLLIN: there's data to read!
  pfd.fd        = xcb_get_file_descriptor(xcb_connection);
  int ntriggers = poll(&pfd, 1, timeout_ms);  // WARN! We CANNOT wait for ntriggers! Otherwise we'll wait processing on events and the screen will go blank because glViewport() will not trigger! Hard to explain, but it happens to me!
  return xcb_poll_for_event(xcb_connection);
}

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

typedef struct{
  i64 nbytes;
  u8* data;
}file_t;

#include <fcntl.h>
file_t* file_init(const char* path){
  file_t* file = malloc(sizeof(file_t));
  int fd = open(path, O_RDONLY);  m_checksys(fd,"open");

  struct stat fs;  stat(path, &fs);
  file->nbytes = fs.st_size;  // NOTE! Follow symlinks, ie. use `fstat` and not `lstat`!
  file->data   = mmap(NULL,file->nbytes, PROT_READ,MAP_SHARED, fd,0);  m_checksys(file->data,"mmap");
  int st = close(fd);                                                  m_checksys(st,"close");
  return file;
}

void file_free(file_t* file){
  int st=munmap(file->data, file->nbytes);  m_checksys(st,"munmap");
  free(file);
}

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#include <fcntl.h>
#include <unistd.h>
void urandom(i64 nbytes, void* data){  int fd;  i64 st;
  fd = open("/dev/urandom", O_RDONLY);
  st = read(fd, data, nbytes);  if(st!=nbytes) printf("\x1b[31mWARN  \x1b[35m%s  \x1b[0m\n", "read");
  st = close(fd);
}

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#define VEC_NELEMS_CAP_INI 256  // WARN! This CANNOT be zero!! =)

#define vec_typedef(ENTRY_TYPE_T, VEC_TYPE_T)  typedef ENTRY_TYPE_T (VEC_TYPE_T)  // The entry type cannot have parentheses! =)
#define vec_get(VEC, IDX)                      ((VEC)[(IDX)])

// ----------------------------------------------------------------
#define vec__nelems_set(VEC, CNT)      (((i64*)(VEC))[-2] = (CNT))  // Don't call these if the vector is an invalid (eg. NULL) pointer!
#define vec__nelems_cap_set(VEC, CAP)  (((i64*)(VEC))[-1] = (CAP))  // Don't call these if the vector is an invalid (eg. NULL) pointer!
#define vec_nelems(    VEC)            ((VEC) ? ((i64*)(VEC))[-2] : (i64)0)
#define vec_nelems_cap(VEC)            ((VEC) ? ((i64*)(VEC))[-1] : (i64)0)
#define vec_nbytes(VEC)                (sizeof(*VEC) * vec_nelems(VEC))
#define vec_nbytes_cap(VEC)            (sizeof(*VEC) * vec_nelems_cap(VEC))

#define vec_meta(VEC)  printf("\x1b[33mMETA\x1b[0m  \x1b[94mvec_t\x1b[0m  \x1b[31mnelems\x1b[0m %'ld\x1b[91m/\x1b[0m%'ld  \x1b[32mnbytes\x1b[0m %'ld\x1b[91m/\x1b[0m%'ld\n", vec_nelems(VEC), vec_nelems_cap(VEC), vec_nbytes(VEC), vec_nbytes_cap(VEC))

// ----------------------------------------------------------------
// @fun `vec_init`.
// This is a mighty GCC macro kind called a `statement expression`! BEST GGC FEATURE EVER!  https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
// The last thing in the compound statement should be an expression followed by a semicolon; the value of this subexpression serves as the value of the entire construct. (If you use some other kind of statement last within the braces, the construct has type void, and thus effectively no value!)
#define vec_init(VEC_TYPE_T)({                                          \
  i64  nbytes = 2*sizeof(i64) + sizeof(VEC_TYPE_T)*VEC_NELEMS_CAP_INI;  \
  i64* base   = malloc(nbytes);                                         \
  vec__nelems_set((void*)(&base[2]), 0);                                \
  vec__nelems_cap_set((void*)(&base[2]), VEC_NELEMS_CAP_INI);           \
  (void*)(&base[2]);  /*Return value!*/                                 \
})

// @fun `vec_reserve`. Ensure that the vector is at least @count elements big
#define vec_reserve(VEC, NVALS){                            \
  i64  nbytes = sizeof(i64)*2  + sizeof(*(VEC)) * (NVALS);  \
  i64* base   = realloc(&((i64*)(VEC))[-2], nbytes);        \
  (VEC) = (void*)(&base[2]);  /*We NEED this assignment!*/  \
  vec__nelems_cap_set((VEC), (NVALS));                      \
}

// ----------------------------------------------------------------
// @fun `vec_push`. Append an element to the end of the vector. The most important function of the whole thing!
#define vec_push(VEC, ENTRY){                   \
  i64 nelems     = vec_nelems((VEC)) + 1;       \
  i64 nelems_cap = vec_nelems_cap((VEC));       \
  if(nelems_cap < nelems){                      \
    vec_reserve((VEC), 2*nelems_cap);           \
  }                                             \
  VEC[vec_nelems((VEC))] = (ENTRY);             \
  vec__nelems_set((VEC), vec_nelems((VEC))+1);  \
}

// @fun `vec_pop`. Delete the last element from the vector!
#define vec_pop(VEC){              \
  i64 nelems = vec_nelems((VEC));  \
  vec_del((VEC), nelems-1);        \
}

// @fun `vec_del`. Delete the element at index @IDX from the vector
#define vec_del(VEC, IDX){              \
  i64 nelems = vec_nelems((VEC));       \
  if((IDX)<nelems){                     \
    vec__nelems_set((VEC), nelems-1);   \
    for(i64 i=(IDX); i<nelems-1; ++i){  \
      (VEC)[i] = (VEC)[i+1];            \
    }                                   \
  }                                     \
}

// @fun `vec_clear`.
#define vec_clear(VEC)  vec__nelems_set(VEC, 0)

// @fun `vec_free`. Free all memory associated with the vector, from the base (which is -2 spots from the actual vector, because it starts at the metadata)!
#define vec_free(VEC){             \
  i64* base = &((i64*)(VEC))[-2];  \
  free(base);                      \
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block  main app! */


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
typedef struct{
  // major dstructs!
  win_t*  win;
  i32     ngpus;
  gpu_t** gpus;
  // user/app/rendering state
  int           running;
  i64           draw_tabs;  // Absolute     time of last draw!
  i64           draw_tdel;  // Differential time of last draw!
  xcb_keycode_t keycode;
}app_t;


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#include <cuda.h>

gpu_t* gpu_init(int gpu_id, win_t* win){  CUresult st;  // NOTE! By convention, in this init function we only call CUDA Driver API that don't require an active CUcontext!
  gpu_t* gpu  = malloc(sizeof(gpu_t));
  gpu->dev.id = gpu_id;

  // ----------------------------------------------------------------
  st=cuDeviceGet(&gpu->dev.dev, gpu->dev.id);                       cu_check(st);
  st=cuDevicePrimaryCtxSetFlags(gpu->dev.dev, CU_CTX_SCHED_YIELD);  cu_check(st);  // CU_CTX_SCHED_SPIN CU_CTX_SCHED_YIELD CU_CTX_SCHED_BLOCKING_SYNC CU_CTX_SCHED_AUTO CU_CTX_MAP_HOST CU_CTX_LMEM_RESIZE_TO_MAX
  st=cuDevicePrimaryCtxRetain(&gpu->dev.ctx, gpu->dev.dev);         cu_check(st);  // Unlike cuCtxCreate(), the newly created context is not pushed onto the stack!

  char gpu_name[33] = {0x00};
  st=cuDeviceGetName(gpu_name, sizeof(gpu_name)-1, gpu->dev.dev);  cu_check(st);
  int ctx_flags,ctx_active; st=cuDevicePrimaryCtxGetState(gpu->dev.dev, &ctx_flags,&ctx_active);  cu_check(st);
  CUcontext ctx_current;    st=cuCtxGetCurrent(&ctx_current);                                     cu_check(st);

  // ----------------------------------------------------------------
  int attr0; st=cuDeviceGetAttribute(&attr0, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,                gpu->dev.dev);  cu_check(st);
  int attr1; st=cuDeviceGetAttribute(&attr1, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,          gpu->dev.dev);  cu_check(st);
  int attr2; st=cuDeviceGetAttribute(&attr2, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,                 gpu->dev.dev);  cu_check(st);
  int attr3; st=cuDeviceGetAttribute(&attr3, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,       gpu->dev.dev);  cu_check(st);
  int attr4; st=cuDeviceGetAttribute(&attr4, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, gpu->dev.dev);  cu_check(st);
  int attr5; st=cuDeviceGetAttribute(&attr5, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,     gpu->dev.dev);  cu_check(st);
  int attr6; st=cuDeviceGetAttribute(&attr6, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,    gpu->dev.dev);  cu_check(st);
  // int attr7; st=cuDeviceGetAttribute(&attr7, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, gpu->dev.dev);
  printf("\n\x1b[32mcuda  \x1b[33m%d\x1b[0m\x1b[91m:\x1b[0m\x1b[92m%s\x1b[91m:\x1b[94m%'d\x1b[91m:\x1b[31m%d\x1b[91m:\x1b[32m%d\x1b[91m:\x1b[31m%d\x1b[91m:\x1b[32m%d\x1b[91m:\x1b[94m%d\x1b[91m:\x1b[97m%d  \x1b[0mctx\x1b[91m-\x1b[35m%08x  \x1b[92m%d\x1b[91m:\x1b[37m%08x  \x1b[0mctx\x1b[91m-\x1b[35m%08x  \x1b[0m\n", gpu->dev.id,gpu_name,attr2, attr3,attr0, attr4,attr1,attr6,attr5, gpu->dev.ctx, ctx_active,ctx_flags, ctx_current);

  // ----------------------------------------------------------------
  st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);
  st=cuStreamCreate(&gpu->dev.stream1, CU_STREAM_NON_BLOCKING);  cu_check(st);

  char* ker_name; char* mod_name;
  mod_name = "pm_pt.cubin";
  st=cuModuleLoad(&gpu->dev.mod_pt, mod_name);  cu_check(st);
  ker_name="ker_light_shader";  st=cuModuleGetFunction(&gpu->dev.ker_light_shader, gpu->dev.mod_pt, ker_name); cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);
  ker_name="ker_mesh0_shader";  st=cuModuleGetFunction(&gpu->dev.ker_mesh0_shader, gpu->dev.mod_pt, ker_name); cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);
  ker_name="ker_pixel_shader";  st=cuModuleGetFunction(&gpu->dev.ker_pixel_shader, gpu->dev.mod_pt, ker_name);  cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);

  mod_name = "pm_bvh.cubin";
  st=cuModuleLoad(&gpu->dev.mod_bvh, mod_name);  cu_check(st);
  ker_name="ker_mesh_radixtree_make";   st=cuModuleGetFunction(&gpu->dev.ker_mesh_radixtree_make,  gpu->dev.mod_bvh, ker_name); cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);
  ker_name="ker_mesh_aabbtree_make";    st=cuModuleGetFunction(&gpu->dev.ker_mesh_aabbtree_make,   gpu->dev.mod_bvh, ker_name); cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);
  ker_name="ker_sort_idxs_by_mortons";  st=cuModuleGetFunction(&gpu->dev.ker_sort_idxs_by_mortons, gpu->dev.mod_bvh, ker_name); cu_check(st);  printf("\x1b[32mCUDA  \x1b[33mload  \x1b[91mkernel \x1b[0m%-24s \x1b[92m@ \x1b[91mmodule \x1b[0m%s  \x1b[0m\n", ker_name, mod_name);

  // --------------------------------------------------------------- TODO! We don't ACTUALLY support arbitrary rectangular tiles yet, but only rectangular tiles satisfying the following properties: 0) each tile row fills a whole img row, and 1) a can begin at any row but only at column zero! More general rectangular tiles will require a custom GPU kernel to do the final copy to the display framebuffer (currently we can manage doing a single cuMemcpy() per tile/GPU)!
  gpu->fb.frame      = 0;
  gpu->fb.nsamples   = PM_NSAMPLES;
  gpu->fb.nbounces   = PM_NBOUNCES;
  // gpu->fb.img_data   = win->cu_data;  // Not ready yet! Is there another way?
  gpu->fb.img_dim_h  = win->dim_h;
  gpu->fb.img_dim_w  = win->dim_w;
  gpu->fb.tile_pos_r = gpu->dev.id * gpu->fb.img_dim_h/2;
  gpu->fb.tile_pos_c = 0;
  gpu->fb.tile_dim_h = gpu->fb.img_dim_h/2;
  gpu->fb.tile_dim_w = gpu->fb.img_dim_w;
  gpu->fb.cam_pos    = (vec3){PM_CAM_POS};
  gpu->fb.cam_dir    = (vec3){PM_CAM_DIR};
  gpu->fb.cam_mov    = (vec3){0,0,0};
  gpu->fb.cam_rot    = (vec3){0,0,0};

  urandom(4, &gpu->fb.seed);
  st=cuMemAlloc((CUdeviceptr*)&gpu->fb.tile_data,  sizeof(u32) *win->xcb_screen->height_in_pixels*win->xcb_screen->width_in_pixels);  cu_check(st);
  st=cuMemAlloc((CUdeviceptr*)&gpu->fb.tile_accum, sizeof(vec3)*win->xcb_screen->height_in_pixels*win->xcb_screen->width_in_pixels);  cu_check(st);  // st=cuMemAllocHost((void**)&data_cpu, nbytes);  cu_check(st);

  // ---------------------------------------------------------------
  st=cuCtxPopCurrent(NULL);  cu_check(st);
  return gpu;
}

void gpu_free(gpu_t* gpu){  CUresult st;
  st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);

  st=cuMemFree((CUdeviceptr)gpu->fb.tile_data);     cu_check(st);
  st=cuMemFree((CUdeviceptr)gpu->fb.tile_accum);    cu_check(st);

  st=cuMemFree((CUdeviceptr)gpu->scene.lights);     cu_check(st);
  st=cuMemFree((CUdeviceptr)gpu->scene.triangles);  cu_check(st);
  st=cuMemFree((CUdeviceptr)gpu->scene.cylinders);  cu_check(st);
  st=cuMemFree((CUdeviceptr)gpu->scene.spheres);    cu_check(st);

  st=cuModuleUnload(gpu->dev.mod_pt);   cu_check(st);
  st=cuModuleUnload(gpu->dev.mod_bvh);  cu_check(st);

  st=cuStreamDestroy(gpu->dev.stream1);  cu_check(st);

  st=cuCtxPopCurrent(NULL);  cu_check(st);
  st=cuDevicePrimaryCtxRelease(gpu->dev.dev);  cu_check(st);  // Unlike cuCtxDestroy(), cuDevicePrimaryCtxRelease() does not pop the context from the stack in any circumstances!
  free(gpu);
}

void gpu_fb_reset(gpu_t* gpu){
  gpu->fb.frame = 0;
  cuMemsetD32Async((CUdeviceptr)gpu->fb.tile_accum, 0x00000000, 3*gpu->fb.tile_dim_h*gpu->fb.tile_dim_w, gpu->dev.stream1);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
void pm_ev_handle(app_t* app){
  f32 wait_secs = m_max(0., 0.016 - (dt_abs()-app->draw_tabs)/1e9);  // printf("wait_secs %d\n", (int)(1000.f*wait_secs));
  xcb_generic_event_t* ev = xcb_ev_poll(app->win->xcb_connection, (int)(1000.f*wait_secs));  if(ev==NULL) return;  // Only process input if there's input to process!

  // ----------------------------------------------------------------------------------------------------------------------------#
  switch(ev->response_type & 0b01111111){
    // ----------------------------------------------------------------------------------------------------------------------------#
    case XCB_KEY_PRESS:{  xcb_key_press_event_t* ev_key_press = (xcb_key_press_event_t*)ev;
      app->keycode = ev_key_press->detail;  // printf("\x1b[94m%02x  \x1b[0m\n", app->keycode);
      switch(app->keycode){
        case 0x09: app->running=0; break;  // This ensures we go through the renderpass clean-up!
        case 0x24: break;
        // ----------------------------------------------------------------
        case 0x18: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x0 -= 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x2a: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x0 += 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x3a: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x1 -= 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x2e: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x1 += 0.01f; gpu_fb_reset(app->gpus[i]); }  break;

        case 0x28: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x2 -= 0.05f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x27: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.mov.x2 += 0.05f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x1c: break;
        case 0x39: break;

        case 0x34: break;
        case 0x35: break;
        case 0x36: break;
        case 0x29: break;
        // ----------------------------------------------------------------
        case 0x1d: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x0 -= 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x1e: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x0 += 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x37: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x1 -= 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x30: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x1 += 0.01f; gpu_fb_reset(app->gpus[i]); }  break;

        case 0x26: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x2 -= 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x1a: for(int i=0; i<app->ngpus; ++i){ app->gpus[i]->scene.rot.x2 += 0.01f; gpu_fb_reset(app->gpus[i]); }  break;
        case 0x20: break;
        case 0x2b: break;

        case 0x21: break;
        case 0x3b: break;
        case 0x3c: break;
        case 0x3d: break;
      }
    }break;
    // ----------------------------------------------------------------------------------------------------------------------------#
    case XCB_CONFIGURE_NOTIFY:{  xcb_configure_notify_event_t* ev_configure_notify = (xcb_configure_notify_event_t*)ev;
      app->win->dim_h = ev_configure_notify->height;
      app->win->dim_w = ev_configure_notify->width;

      for(int i=0; i<app->ngpus; ++i){
        app->gpus[i]->fb.img_dim_h  = app->win->dim_h;
        app->gpus[i]->fb.img_dim_w  = app->win->dim_w;
        app->gpus[i]->fb.tile_pos_r = app->gpus[i]->dev.id * app->gpus[i]->fb.img_dim_h/2;
        app->gpus[i]->fb.tile_pos_c = 0;
        app->gpus[i]->fb.tile_dim_h = app->gpus[i]->fb.img_dim_h/2;
        app->gpus[i]->fb.tile_dim_w = app->gpus[i]->fb.img_dim_w;
        gpu_fb_reset(app->gpus[i]);
      }
    }break;
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  free(ev);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
void pm_draw(app_t* app){  CUresult st;  // NOTE! We only draw if the time since the last draw is at least 8ms (or 16ms, or something like that)!
  // printf("lala %lld  meme %lld\n", dt_abs()-app->draw_tabs, app->draw_tdel);
  if((dt_abs()-app->draw_tabs)/1e9 < 0.016){ /*printf("skip \x1b[32m%.6f  \x1b[0m\n", (dt_abs()-app->draw_tabs)/1e9);*/ return; }
  printf("\x1b[0mtime \x1b[32m%.6f  ", (dt_abs()-app->draw_tabs)/1e9);
  app->draw_tabs = dt_abs();
  app->draw_tdel = dt_abs();

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 1) mesh & bvh shaders!
  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);

    void* args_light_shader[] = {&gpu->fb, &gpu->scene};
    st=cuLaunchKernel(gpu->dev.ker_light_shader, 1,1,1, 1,1,1, 0,gpu->dev.stream1, args_light_shader,NULL);  cu_check(st);

    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.aabbs,          0b00000000, sizeof(aabb_t)           * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);  // Reset the BVH data Justin Case!
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.mortons,        0b00000000, sizeof(u64)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.idxs,           0b00000000, sizeof(u32)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_nodes,     0b00000000, sizeof(struct bvh_node_t)*(gpu->scene.mesh0.nelems-1), gpu->dev.stream1);  cu_check(st);  // For internal nodes, @leaf is 0
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_leaves,    0b11111111, sizeof(struct bvh_node_t)* gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);  // For leaves, @leaf is 1
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_semaphore, 0b00000000, sizeof(i32)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);

    void* args_mesh0_shader[] = {&gpu->fb, &gpu->scene};
    st=cuLaunchKernel(gpu->dev.ker_mesh0_shader, m_divceilu(gpu->scene.mesh0.nelems,PM_GPU_THR_LVL0_2DIM_X*PM_GPU_THR_LVL0_2DIM_Y),1,1, PM_GPU_THR_LVL0_2DIM_X*PM_GPU_THR_LVL0_2DIM_Y,1,1, 0,gpu->dev.stream1, args_mesh0_shader,NULL);  cu_check(st);

    i32 ev_triangles_radixtree_nnodes = gpu->scene.mesh0.nelems-1;
    void* args_mesh0_radixtree_make[] = {&ev_triangles_radixtree_nnodes, &gpu->scene.mesh0.tree_nodes,&gpu->scene.mesh0.tree_leaves};
    st=cuLaunchKernel(gpu->dev.ker_mesh_radixtree_make, ((u32)ev_triangles_radixtree_nnodes+BVH_NTHREADS-1)/BVH_NTHREADS,1,1, BVH_NTHREADS,1,1, 0,gpu->dev.stream1, args_mesh0_radixtree_make,NULL);  cu_check(st);  // Make a radix tree, which the actual BVH needs! Described in karras2012 paper!

#if 0  // BUG! Sorting the primitives actually performes WORSE!  #include <cub/device/device_radix_sort.cuh>  -lstdc++
    void* mesh_data=0;  size_t mesh_data_nbytes;
    cub::DeviceRadixSort::SortPairs(mesh_data,mesh_data_nbytes, gpu->scene.mesh0.mortons,gpu->scene.mesh0.mortons, gpu->scene.mesh0.idxs,gpu->scene.mesh0.idxs, gpu->scene.mesh0.nelems);  cudaMalloc(&mesh_data, mesh_data_nbytes);  // Determine storage requirements!
    cub::DeviceRadixSort::SortPairs(mesh_data,mesh_data_nbytes, gpu->scene.mesh0.mortons,gpu->scene.mesh0.mortons, gpu->scene.mesh0.idxs,gpu->scene.mesh0.idxs, gpu->scene.mesh0.nelems);  cudaFree(mesh_data);                       // NOW run sorting operation...
#endif
    // TODO! Do these launch parameters seem sane? Shouldn't it be `(nelems + BVH_NTHREADS-1) / BVH_NTHREADS`?
    void* args_mesh_aabbtree_make[] = {&gpu->scene.mesh0.nelems,&gpu->scene.mesh0.aabbs,&gpu->scene.mesh0.idxs, &gpu->scene.mesh0.tree_nodes,&gpu->scene.mesh0.tree_leaves,&gpu->scene.mesh0.tree_semaphore};
    st=cuLaunchKernel(gpu->dev.ker_mesh_aabbtree_make, (u32)gpu->scene.mesh0.nelems+(BVH_NTHREADS-1)/BVH_NTHREADS,1,1, BVH_NTHREADS,1,1, 0,gpu->dev.stream1, args_mesh_aabbtree_make,NULL);  cu_check(st);  // Described in karras2012 paper!
    st=cuCtxPopCurrent(NULL);  cu_check(st);
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 2) sync?  TODO! Do we need this?
#if 0
  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    st=cuCtxPushCurrent(gpu->dev.ctx);         cu_check(st);
    st=cuStreamSynchronize(gpu->dev.stream1);  cu_check(st);
    st=cuCtxPopCurrent(NULL);  cu_check(st);
  }
#endif

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 3) pixel shader!
  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);
    void* ker_args[] = {&gpu->fb, &gpu->scene};
    st=cuLaunchKernel(gpu->dev.ker_pixel_shader, m_divceilu(gpu->fb.tile_dim_w,PM_GPU_THR_LVL0_2DIM_X),m_divceilu(gpu->fb.tile_dim_h,PM_GPU_THR_LVL0_2DIM_Y),1, PM_GPU_THR_LVL0_2DIM_X,PM_GPU_THR_LVL0_2DIM_X,1, 0,gpu->dev.stream1, ker_args,NULL);  cu_check(st);
    st=cuMemcpyAsync((CUdeviceptr)gpu->fb.img_data + sizeof(u32)*gpu->fb.img_dim_w * gpu->fb.tile_pos_r, (CUdeviceptr)gpu->fb.tile_data, sizeof(u32)*gpu->fb.tile_dim_h*gpu->fb.tile_dim_w, gpu->dev.stream1);  cu_check(st);
    st=cuCtxPopCurrent(NULL);  cu_check(st);
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 4) sync?  TODO! Do we need this?
#if 1
  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);
    st=cuStreamSynchronize(gpu->dev.stream1);  cu_check(st);
    st=cuCtxPopCurrent(NULL);  cu_check(st);
  }
#endif

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 5) blit from the CUDA framebuffer to the OpenGL display framebuffer!
  glTexSubImage2D(GL_TEXTURE_RECTANGLE,0, 0,0,app->win->dim_w,app->win->dim_h, GL_RGBA,GL_UNSIGNED_BYTE, NULL);
  glBlitFramebuffer(0,0,app->win->dim_w,app->win->dim_h, 0,0,app->win->dim_w,app->win->dim_h, GL_COLOR_BUFFER_BIT,GL_NEAREST);  // When the color buffer is transferred, values are taken from the read buffer of the specified read framebuffer and written to each of the draw buffers of the specified DRAW_FRAMEBUFFER.
  glFlush();
  app->draw_tdel = dt_abs() - app->draw_tdel;  printf("\x1b[0mdraw \x1b[32m%.6f  \x1b[0m\n", app->draw_tdel/1e9);

  // ----------------------------------------------------------------------------------------------------------------------------#
  // 6) update app state!
  for(int i=0; i<app->ngpus; ++i){
    app->gpus[i]->fb.frame += 1;
    app->gpus[i]->fb.seed   = (app->gpus[i]->fb.seed + app->gpus[i]->fb.seed*0x5851f42du) ^ app->gpus[i]->fb.seed;
    app->gpus[i]->scene.mov = (vec3){0,0,0};  // Consume it!
    app->gpus[i]->scene.rot = (vec3){0,0,0};  // Consume it!
  }
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block */
#define PDB_CASE_NONE    0b00000000  // 2**-inf
#define PDB_CASE_HEADER  (1ull<<0x00)  // Mandatory.
#define PDB_CASE_OBSLTE  (1ull<<0x01)  // Optional.  Mandatory in entries that have been replaced by a newer entry.
#define PDB_CASE_TITLE   (1ull<<0x02)  // Mandatory.
#define PDB_CASE_SPLIT   (1ull<<0x03)  // Optional.  Mandatory when large macromolecular complexes are split into multiple PDB entries.
#define PDB_CASE_CAVEAT  (1ull<<0x04)  // Optional.  Mandatory when there are outstanding errors such as chirality.
#define PDB_CASE_COMPND  (1ull<<0x05)  // Mandatory.
#define PDB_CASE_SOURCE  (1ull<<0x06)  // Mandatory.
#define PDB_CASE_KEYWDS  (1ull<<0x07)  // Mandatory.
#define PDB_CASE_EXPDTA  (1ull<<0x08)  // Mandatory.
#define PDB_CASE_NUMMDL  (1ull<<0x09)  // Optional.  Mandatory for NMR ensemble entries.
#define PDB_CASE_MDLTYP  (1ull<<0x0a)  // Optional.  Mandatory for NMR minimized average structures or when the entire polymer chain contains C alpha or P atoms only.
#define PDB_CASE_AUTHOR  (1ull<<0x0b)  // Mandatory.
#define PDB_CASE_REVDAT  (1ull<<0x0c)  // Mandatory.
#define PDB_CASE_SPRSDE  (1ull<<0x0d)  // Optional.  Mandatory for a replacement entry.
#define PDB_CASE_JRNL    (1ull<<0x0e)  // Optional.  Mandatory for a publication describes the experiment.
#define PDB_CASE_REMARK0 (1ull<<0x0f)  // Optional.  Mandatory for a re-refined structure.
#define PDB_CASE_REMARK1 (1ull<<0x10)  // Optional.
#define PDB_CASE_REMARK2 (1ull<<0x11)  // Mandatory.
#define PDB_CASE_REMARK3 (1ull<<0x12)  // Mandatory.
#define PDB_CASE_REMARKN (1ull<<0x13)  // Optional.  Mandatory under certain conditions.
#define PDB_CASE_DBREF   (1ull<<0x14)  // Optional.  Mandatory for all polymers.
#define PDB_CASE_DBREF1  (1ull<<0x15)  // Optional.  Mandatory when certain sequence database accession and/or sequence numbering does not fit preceding DBREF format.
#define PDB_CASE_DBREF2  (1ull<<0x16)  // Optional.  Mandatory when certain sequence database accession and/or sequence numbering does not fit preceding DBREF format.
#define PDB_CASE_SEQADV  (1ull<<0x17)  // Optional.  Mandatory if sequence conflict exists.
#define PDB_CASE_SEQRES  (1ull<<0x18)  // Mandatory. Mandatory if ATOM records exist.
#define PDB_CASE_MODRES  (1ull<<0x19)  // Optional.  Mandatory if modified group exists in the coordinates.
#define PDB_CASE_HET     (1ull<<0x1a)  // Optional.  Mandatory if a non-standard group other than water appears in the coordinates.
#define PDB_CASE_HETNAM  (1ull<<0x1b)  // Optional.  Mandatory if a non-standard group other than water appears in the coordinates.
#define PDB_CASE_HETSYN  (1ull<<0x1c)  // Optional.
#define PDB_CASE_FORMUL  (1ull<<0x1d)  // Optional.  Mandatory if a non-standard group or water appears in the coordinates.
#define PDB_CASE_HELIX   (1ull<<0x1e)  // Optional.
#define PDB_CASE_SHEET   (1ull<<0x1f)  // Optional.
#define PDB_CASE_SSBOND  (1ull<<0x20)  // Optional.  Mandatory if a disulfide bond is present.
#define PDB_CASE_LINK    (1ull<<0x21)  // Optional.  Mandatory if non-standard residues appear in a polymer.
#define PDB_CASE_CISPEP  (1ull<<0x22)  // Optional.
#define PDB_CASE_SITE    (1ull<<0x23)  // Optional.
#define PDB_CASE_CRYST1  (1ull<<0x24)  // Mandatory.
#define PDB_CASE_ORIGX1  (1ull<<0x25)  // Mandatory.
#define PDB_CASE_ORIGX2  (1ull<<0x26)  // Mandatory.
#define PDB_CASE_ORIGX3  (1ull<<0x27)  // Mandatory.
#define PDB_CASE_SCALE1  (1ull<<0x28)  // Madatory.
#define PDB_CASE_SCALE2  (1ull<<0x29)  // Mandatory.
#define PDB_CASE_SCALE3  (1ull<<0x2a)  // Mandatory.
#define PDB_CASE_MTRIX1  (1ull<<0x2b)  // Optional.  Mandatory if the complete asymmetric unit must be generated from the given coordinates using non-crystallographic symmetry.
#define PDB_CASE_MTRIX2  (1ull<<0x2c)  // Optional.  Mandatory if the complete asymmetric unit must be generated from the given coordinates using non-crystallographic symmetry.
#define PDB_CASE_MTRIX3  (1ull<<0x2d)  // Optional.  Mandatory if the complete asymmetric unit must be generated from the given coordinates using non-crystallographic symmetry.
#define PDB_CASE_MODEL   (1ull<<0x2e)  // Optional.  Mandatory if more than one model present in the entry.
#define PDB_CASE_ATOM    (1ull<<0x2f)  // Optional.  Mandatory if standard residues exist.
#define PDB_CASE_ANISOU  (1ull<<0x30)  // Optional.
#define PDB_CASE_TER     (1ull<<0x31)  // Optional.  Mandatory if ATOM records exist.
#define PDB_CASE_HETATM  (1ull<<0x32)  // Optional.  Mandatory if non-standard group exists.
#define PDB_CASE_ENDMDL  (1ull<<0x33)  // Optional.  Mandatory if MODEL appears.
#define PDB_CASE_CONECT  (1ull<<0x34)  // Optional.  Mandatory if non-standard group appears and if LINK or SSBOND records exist.
#define PDB_CASE_MASTER  (1ull<<0x35)  // Mandatory.
#define PDB_CASE_END     (1ull<<0x36)  // Mandatory.

static const f32 LUT_F32_EXP[] = {1e-0f,1e-1f,1e-2f,1e-3f,1e-4f,1e-5f,1e-6f};  // 1e-7f,1e-9f,1e-10f,1e-11f,1e-12f  // f32 precision: 7 decimal digits on avg. 89.27% of the range has 7 decimal digits, 10.10% has 8 decimal digits, and 0.63% has 6 decimal digits.
#define asciidec_to_int(BYTE)  ((BYTE) - 0x30)

// ----------------------------------------------------------------------------------------------------------------------------#
// TODO! Using an ascii `?` to find the Cartesian coordinates is THE WORST IDEA EVER!
void* pdbx_parse(const char* path){  CUresult st;
  dt_t dt; dt_ini(&dt);
  file_t*   file    = file_init(path);
  i64       nbytes  = file->nbytes;
  u8*       data    = file->data;
  sphere_t* spheres = vec_init(sphere_t);

  // ----------------------------------------------------------------------------------------------------------------------------#
  while(nbytes>0){
    // NOTE! These comparisons are ONLY VALID AT THE BEGINNING OF A LINE!
    u64 state = 
      (memcmp(data, "ATOM",   4)==0) * PDB_CASE_ATOM   |
      (memcmp(data, "TER",    3)==0) * PDB_CASE_TER    |
      (memcmp(data, "HETATM", 6)==0) * PDB_CASE_HETATM |
      (memcmp(data, "END",    3)==0) * PDB_CASE_END;
    // printf("state %016lx %d %x\n", state, __builtin_popcountll(state), __builtin_ctzll(state));  if(__builtin_popcountll(state)>1) exit(1);

    switch(state){  // Deterministic for-loops are faster than random while-loops!
      // ----------------------------------------------------------------
      case PDB_CASE_ATOM:{
        f32 sign;
        int dec_cnt;
        f32 pos_x0 = 0.f;
        f32 pos_x1 = 0.f;
        f32 pos_x2 = 0.f;
        f32 rad    = 0.f;

        while(data[0] != 0x3f/*?*/){                                                            ++data;--nbytes;  }  ++data;--nbytes;  // printf("\x1b[33matom  \x1b[37m%.*s  \x1b[0m", 34,data);

        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x0*=10.f; pos_x0+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x0+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x0*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x1*=10.f; pos_x1+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x1+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x1*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x2*=10.f; pos_x2+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x2+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x2*=sign;

        // data+=5; nbytes-=5;
        // sign=+1; dec_cnt=0;
        // while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        // while(data[0] != 0x2e/*.*/){  rad*=10.f; rad+=asciidec_to_int(data[0]);                 ++data;--nbytes;  }  ++data;--nbytes;
        // while(data[0] != 0x20/* */){  rad+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];     ++data;--nbytes;  }  ++data;--nbytes;

        // printf("\x1b[31m%7.3f \x1b[32m%7.3f \x1b[94m%7.3f  \x1b[35m%7.3f  \x1b[0m\n", pos_x0,pos_x1,pos_x2);
        sphere_t sphere = {(vec3){pos_x0,pos_x1,pos_x2}, rad, PM_RGB_BLUE};
        vec_push(spheres, sphere);
      }break;
      // ----------------------------------------------------------------
      case PDB_CASE_TER:{  // puts("\x1b[91mter\x1b[0m");
      }break;
      // ----------------------------------------------------------------
      case PDB_CASE_HETATM:{
        f32 sign;
        int dec_cnt;
        f32 pos_x0 = 0.f;
        f32 pos_x1 = 0.f;
        f32 pos_x2 = 0.f;
        f32 rad    = 0.f;

        while(data[0] != 0x3f/*?*/){                                                            ++data;--nbytes;  }  ++data;--nbytes;  // printf("\x1b[33mhetatm  \x1b[37m%.*s  \x1b[0m", 34,data);

        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x0*=10.f; pos_x0+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x0+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x0*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x1*=10.f; pos_x1+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x1+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x1*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x2*=10.f; pos_x2+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x2+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x2*=sign;

        // data+=5; nbytes-=5;
        // sign=+1; dec_cnt=0;
        // while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        // while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        // while(data[0] != 0x2e/*.*/){  rad*=10.f; rad+=asciidec_to_int(data[0]);                 ++data;--nbytes;  }  ++data;--nbytes;
        // while(data[0] != 0x20/* */){  rad+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];     ++data;--nbytes;  }  ++data;--nbytes;

        // printf("\x1b[31m%7.3f \x1b[32m%7.3f \x1b[94m%7.3f  \x1b[35m%7.3f  \x1b[0m\n", pos_x0,pos_x1,pos_x2);
        sphere_t sphere = {(vec3){pos_x0,pos_x1,pos_x2}, rad, PM_RGB_RED};
        vec_push(spheres, sphere);
      }break;
    }

    while(data[0] != 0x0a && nbytes>0){  ++data;--nbytes;  }  // Advance to next line!
    ++data;--nbytes;  // printf("END LINE  %'ld  %'ld  %02x\n", data-file->data, nbytes, data[0]);
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  file_free(file);
  i64 nspheres = vec_nelems(spheres);

  f32 pos_min = +1e38;
  f32 pos_max = -1e38;
  f32 rad_max = -1e38;
  for(int i=0; i<nspheres; ++i){
    sphere_t sphere = spheres[i];
    pos_min = m_min(m_min(m_min(pos_min, spheres[i].pos.x0), spheres[i].pos.x1), spheres[i].pos.x2);
    pos_max = m_max(m_max(m_max(pos_max, spheres[i].pos.x0), spheres[i].pos.x1), spheres[i].pos.x2);
    rad_max = m_max(rad_max, spheres[i].radius);
  }
  for(int i=0; i<nspheres; ++i){
    spheres[i].pos.x0 = 2.f*(spheres[i].pos.x0-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].pos.x1 = 2.f*(spheres[i].pos.x1-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].pos.x2 = 2.f*(spheres[i].pos.x2-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].radius = 0.02f;  // 0.03f * spheres[i].radius/rad_max;
    // printf("sphere  %.3f %.3f %.3f  %.3f  \n", spheres[i].pos.x0,spheres[i].pos.x1,spheres[i].pos.x2, spheres[i].radius);
  }
  dt_end(&dt);
  printf("\nnspheres \x1b[94m%'ld  \x1b[0mparse \x1b[32m%.6f  \x1b[0mpath \x1b[92m%s  \x1b[0m\n", nspheres,dt_del(&dt),path);

  return spheres;
}

// ----------------------------------------------------------------------------------------------------------------------------#
void* pdb_parse(const char* path){  CUresult st;
  dt_t dt; dt_ini(&dt);
  file_t*   file     = file_init(path);
  i64       nbytes   = file->nbytes;
  u8*       data     = file->data;
  sphere_t* spheres  = vec_init(sphere_t);
  int       rgbs_idx = 0;
  int       rgbs_nelems = sizeof(PM_RGBS) / sizeof(PM_RGBS[0]);

  // ----------------------------------------------------------------------------------------------------------------------------#
  while(nbytes>0){
    // NOTE! These comparisons are ONLY VALID AT THE BEGINNING OF A LINE!
    u64 state = 
      (memcmp(data, "ATOM",   4)==0) * PDB_CASE_ATOM   |
      (memcmp(data, "TER",    3)==0) * PDB_CASE_TER    |
      (memcmp(data, "HETATM", 6)==0) * PDB_CASE_HETATM |
      (memcmp(data, "END",    3)==0) * PDB_CASE_END;
    // printf("state %016lx %d %x\n", state, __builtin_popcountll(state), __builtin_ctzll(state));  if(__builtin_popcountll(state)>1) exit(1);

    switch(state){  // Deterministic for-loops are faster than random while-loops!
      // ----------------------------------------------------------------
      case PDB_CASE_ATOM:{
        f32 sign;
        int dec_cnt;
        f32 pos_x0 = 0.f;
        f32 pos_x1 = 0.f;
        f32 pos_x2 = 0.f;
        f32 rad    = 0.f;

        data+=30;nbytes-=30;  // printf("\x1b[33matom  \x1b[37m%.*s  \x1b[0m", 34,data);

        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x0*=10.f; pos_x0+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x0+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x0*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x1*=10.f; pos_x1+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x1+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x1*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x2*=10.f; pos_x2+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x2+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x2*=sign;

        // printf("\x1b[31m%7.3f \x1b[32m%7.3f \x1b[94m%7.3f  \x1b[35m%7.3f  \x1b[0m\n", pos_x0,pos_x1,pos_x2);
        sphere_t sphere = {(vec3){pos_x0,pos_x1,pos_x2}, rad, PM_RGBS[rgbs_idx]};
        vec_push(spheres, sphere);
      }break;
      // ----------------------------------------------------------------
      case PDB_CASE_TER:{  // puts("\x1b[91mter\x1b[0m");
        rgbs_idx = (rgbs_idx+1) % rgbs_nelems;
      }break;
      // ----------------------------------------------------------------
      case PDB_CASE_HETATM:{
        f32 sign;
        int dec_cnt;
        f32 pos_x0 = 0.f;
        f32 pos_x1 = 0.f;
        f32 pos_x2 = 0.f;
        f32 rad    = 0.f;

        data+=30;nbytes-=30;  // printf("\x1b[33mhetatm  \x1b[37m%.*s  \x1b[0m", 34,data);

        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x0*=10.f; pos_x0+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x0+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x0*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x1*=10.f; pos_x1+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x1+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x1*=sign;
        sign=+1; dec_cnt=0;
        while(data[0] == 0x20/* */){                                                            ++data;--nbytes;  }
        while(data[0] == 0x2d/*-*/){  sign*=-1;                                                 ++data;--nbytes;  }
        while(data[0] != 0x2e/*.*/){  pos_x2*=10.f; pos_x2+=asciidec_to_int(data[0]);           ++data;--nbytes;  }  ++data;--nbytes;
        while(data[0] != 0x20/* */){  pos_x2+=asciidec_to_int(data[0])*LUT_F32_EXP[++dec_cnt];  ++data;--nbytes;  }  ++data;--nbytes;  pos_x2*=sign;

        // printf("\x1b[31m%7.3f \x1b[32m%7.3f \x1b[94m%7.3f  \x1b[35m%7.3f  \x1b[0m\n", pos_x0,pos_x1,pos_x2);
        sphere_t sphere = {(vec3){pos_x0,pos_x1,pos_x2}, rad, PM_RGBS[rgbs_idx]};
        vec_push(spheres, sphere);
      }break;
    }

    while(data[0] != 0x0a && nbytes>0){  ++data;--nbytes;  }  // Advance to next line!
    ++data;--nbytes;  // printf("END LINE  %'ld  %'ld  %02x\n", data-file->data, nbytes, data[0]);
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  file_free(file);
  i64 nspheres = vec_nelems(spheres);

  f32 pos_min = +1e38;
  f32 pos_max = -1e38;
  f32 rad_max = -1e38;
  for(int i=0; i<nspheres; ++i){
    sphere_t sphere = spheres[i];
    pos_min = m_min(m_min(m_min(pos_min, spheres[i].pos.x0), spheres[i].pos.x1), spheres[i].pos.x2);
    pos_max = m_max(m_max(m_max(pos_max, spheres[i].pos.x0), spheres[i].pos.x1), spheres[i].pos.x2);
    rad_max = m_max(rad_max, spheres[i].radius);
  }
  for(int i=0; i<nspheres; ++i){
    spheres[i].pos.x0 = 2.f*(spheres[i].pos.x0-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].pos.x1 = 2.f*(spheres[i].pos.x1-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].pos.x2 = 2.f*(spheres[i].pos.x2-pos_min)/(pos_max-pos_min)-1.f;
    spheres[i].radius = 0.02f;  // 0.03f * spheres[i].radius/rad_max;
    // printf("sphere  %.3f %.3f %.3f  %.3f  \n", spheres[i].pos.x0,spheres[i].pos.x1,spheres[i].pos.x2, spheres[i].radius);
  }
  dt_end(&dt);
  printf("\nnspheres \x1b[94m%'ld  \x1b[0mparse \x1b[32m%.6f  \x1b[0mpath \x1b[92m%s  \x1b[0m\n", nspheres,dt_del(&dt),path);

  return spheres;
}




// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @block */
int main(int nargs, char* args[]){  setlocale(LC_NUMERIC,"");
  char* protein_path = PM_PDB_PATH_DEFAULT;  if(nargs>1) protein_path = args[1];
  int   access_st    = access(protein_path, F_OK|R_OK);
  m_checksys(access_st,"access");  // We call this so that we get an error message!
  if(access_st==-1) exit(EXIT_FAILURE);

  // ----------------------------------------------------------------------------------------------------------------------------#
  app_t* app = malloc(sizeof(app_t));
  app->win   = win_init(PM_WIN_DIM_H,PM_WIN_DIM_W);

  CUresult st=cuInit(0);  cu_check(st);  // We ALWAYS call this ONCE and with an arg of ZERO, regardless of the number of GPUs, apparently!
  int version;
  st=cuDriverGetVersion(&version);   cu_check(st);
  st=cuDeviceGetCount(&app->ngpus);  cu_check(st);  printf("\x1b[32mcuda  \x1b[35mcuda_init  \x1b[0mv\x1b[92m%d  \x1b[0mngpus \x1b[32m%d  \x1b[0m\n", version,app->ngpus);

  // ----------------------------------------------------------------
  app->gpus = malloc(sizeof(gpu_t*) * app->ngpus);  // Semantics: an array of pointers!
  for(int i=0; i<app->ngpus; ++i)
    app->gpus[i] = gpu_init(i, app->win);

  // ----------------------------------------------------------------
  win_init2(app->win, app->gpus[PM_GPU_MAIN]->dev.ctx);

  for(int i=0; i<app->ngpus; ++i)
    app->gpus[i]->fb.img_data = app->win->cu_data;

  // ----------------------------------------------------------------------------------------------------------------------------#
  sphere_t* atoms;
  int is_pdb = memcmp(protein_path + strlen(protein_path)-strlen(".pdb"), ".pdb", strlen(".pdb"))==0;
  if(is_pdb)  atoms = pdb_parse(protein_path);
  else        atoms = pdbx_parse(protein_path);  // Else assume it's PDBx/mmCIF!

  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    gpu->scene.nlights    = PM_NLIGHTS;
    gpu->scene.ntriangles = 0;
    gpu->scene.ncylinders = 0;
    gpu->scene.nspheres   = vec_nelems(atoms);
    gpu->scene.nelems     = gpu->scene.nlights + gpu->scene.ntriangles + gpu->scene.ncylinders + gpu->scene.nspheres;
    gpu->scene.mov        = (vec3){0,0,0};
    gpu->scene.rot        = (vec3){0,0,0};

    gpu->scene.mesh0.gtype  = GTYPE_SPHERE;
    gpu->scene.mesh0.nelems = vec_nelems(atoms);
    gpu->scene.mesh0.mov    = gpu->scene.mov;
    gpu->scene.mesh0.rot    = gpu->scene.rot;

    st=cuCtxPushCurrent(gpu->dev.ctx);  cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.lights,    sizeof(light_t)   *m_max(1,gpu->scene.nlights));     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.triangles, sizeof(triangle_t)*m_max(1,gpu->scene.ntriangles));  cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.cylinders, sizeof(cylinder_t)*m_max(1,gpu->scene.ncylinders));  cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.spheres,   sizeof(sphere_t)  *m_max(1,gpu->scene.nspheres));    cu_check(st);

    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.data,          sizeof(sphere_t)         * gpu->scene.mesh0.nelems);     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.aabbs,         sizeof(aabb_t)           * gpu->scene.mesh0.nelems);     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.mortons,       sizeof(u64)              * gpu->scene.mesh0.nelems);     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.idxs,          sizeof(u32)              * gpu->scene.mesh0.nelems);     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.tree_nodes,    sizeof(struct bvh_node_t)*(gpu->scene.mesh0.nelems-1));  cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.tree_leaves,   sizeof(struct bvh_node_t)* gpu->scene.mesh0.nelems);     cu_check(st);
    st=cuMemAlloc((CUdeviceptr*)&gpu->scene.mesh0.tree_semaphore,sizeof(i32)              * gpu->scene.mesh0.nelems);     cu_check(st);

    st=cuMemcpyHtoDAsync((CUdeviceptr)gpu->scene.spheres,    atoms, sizeof(sphere_t)*gpu->scene.nspheres,     gpu->dev.stream1);  cu_check(st);
    st=cuMemcpyHtoDAsync((CUdeviceptr)gpu->scene.mesh0.data, atoms, sizeof(sphere_t)*gpu->scene.mesh0.nelems, gpu->dev.stream1);  cu_check(st);

    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.aabbs,          0b00000000, sizeof(aabb_t)           * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);  // Reset the BVH data Justin Case!
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.mortons,        0b00000000, sizeof(u64)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.idxs,           0b00000000, sizeof(u32)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_nodes,     0b00000000, sizeof(struct bvh_node_t)*(gpu->scene.mesh0.nelems-1), gpu->dev.stream1);  cu_check(st);  // For internal nodes, @leaf is 0
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_leaves,    0b11111111, sizeof(struct bvh_node_t)* gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);  // For leaves, @leaf is 1
    st=cuMemsetD8Async((CUdeviceptr)gpu->scene.mesh0.tree_semaphore, 0b00000000, sizeof(i32)              * gpu->scene.mesh0.nelems,    gpu->dev.stream1);  cu_check(st);

    st=cuCtxPopCurrent(NULL);  cu_check(st);
  }

  for(int i=0; i<app->ngpus; ++i){
    gpu_t* gpu = app->gpus[i];
    st=cuCtxPushCurrent(gpu->dev.ctx);         cu_check(st);
    st=cuStreamSynchronize(gpu->dev.stream1);  cu_check(st);
    st=cuCtxPopCurrent(NULL);                  cu_check(st);
  }

  vec_free(atoms);

  // ----------------------------------------------------------------------------------------------------------------------------#
  putchar(0x0a);
  app->running   = 1;
  app->draw_tabs = dt_abs();
  app->draw_tdel = dt_abs();

  while(app->running==1){
    pm_ev_handle(app);  // printf("%.3f %.3f %.3f\n", app->gpus[0]->scene.mov.x0,app->gpus[0]->scene.mov.x1,app->gpus[0]->scene.mov.x2);
    pm_draw(app);  // Render pass!
  }

  // ----------------------------------------------------------------------------------------------------------------------------#
  win_free(app->win);

  for(int i=0; i<app->ngpus; ++i)
    gpu_free(app->gpus[i]);
  free(app->gpus);

  free(app);
  puts("bye!");
  exit(EXIT_SUCCESS);
}
