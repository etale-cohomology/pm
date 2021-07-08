// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
#define M_TAU  6.28

#define PM_GPU_MAIN             1
#define PM_EPSILON              0.001f
#define PM_GPU_THR_LVL0_2DIM_X  8
#define PM_GPU_THR_LVL0_2DIM_Y  8

#define PM_NLIGHTS    6     // The number of 2-faces of a cube, since we want a "lightbox", ie. a box of lights!
#define PM_NSAMPLES   1     // yt:1<<6  // (starts converging at about 1<<10 samples)
#define PM_NBOUNCES   4     // yt:4     // (starts converging at about 6 bounces)
#define PM_WIN_DIM_H  (32*11)  // yt:1024  // 1920*1080 / ((32*11)**2 / 2)
#define PM_WIN_DIM_W  (32*11)  // yt:1024  // 1920*1080 / ((32*11)**2 / 2)

#define PM_CAM_FOV    (3.141592653589793238f / 3)  // pi/2: 90 fov, pi/3: 60 fov, pi/4: 45 fov, pi/6: 30 fov
#define PM_CAM_POS    { 0.2, 0.1, 6.0}  // 3.8 6
#define PM_CAM_DIR    {-0.0,-0.0,-1.0}

#define PM_PDB_PATH_DEFAULT  "/mnt/ssd0/bio/d0/6vxx.cif"

#define PM_RGB_RED       0xff6666
#define PM_RGB_GREEN     0x44bb99  // 0x90ee90 0x98fb98
#define PM_RGB_BLUE      0x0099ff
#define PM_RGB_PURPLE    0x8866dd  // 0xa29bfe
#define PM_RGB_YELLOW    0xffff66  // e7e77e
#define PM_RGB_AMARANTH  0xff3366
#define PM_RGB_PINK      0xffaabb
#define PM_RGB_MAGENTA   0xff88ff
#define PM_RGB_INDIGO    0x6677bb
#define PM_RGB_CYAN      0x26c6da
#define PM_RGB_TEAL      0x26a69a
#define PM_RGB_LIME      0xccdd55
#define PM_RGB_AMBER     0xffcc22
#define PM_RGB_ORANGE    0xffaa22
#define PM_RGB_BROWN     0x8d6e63
uint PM_RGBS[] = {PM_RGB_BLUE,PM_RGB_GREEN,PM_RGB_RED,PM_RGB_PURPLE,PM_RGB_YELLOW,PM_RGB_AMARANTH,PM_RGB_PINK,PM_RGB_MAGENTA,PM_RGB_INDIGO,PM_RGB_CYAN,PM_RGB_TEAL,PM_RGB_LIME,PM_RGB_AMBER,PM_RGB_ORANGE,PM_RGB_BROWN};

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  types: low-level! */
#include <stdint.h>

typedef  float     f32;
typedef  double    f64;
typedef  int8_t    i8;
typedef  int16_t   i16;
typedef  int32_t   i32;
typedef  int64_t   i64;
typedef  uint8_t   u8;
typedef  uint16_t  u16;
typedef  uint32_t  u32;
typedef  uint64_t  u64;

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  types: mid-level! */
typedef union{
  f32 d[3];
  struct{ f32 x0, x1, x2; };
}vec3;

typedef union{
  f32 d[4];
  struct{ f32 x0, x1, x2, x3; };
}quat;

// ----------------------------------------------------------------
enum gtype_t{GTYPE_UNKNOWN, GTYPE_LIGHT, GTYPE_TRIANGLE, GTYPE_CYLINDER, GTYPE_SPHERE};  // Geometric type!

typedef struct{
  vec3 min, max;
}aabb_t;

typedef struct{
  vec3 vert0, edge01, edge02;
  vec3 emission;
}light_t;

typedef struct{
  vec3 vert0, edge01, edge02;
  u32  albedo;
}triangle_t;

typedef struct{
  vec3 pos, dir;
  f32  radius, height;
  u32  albedo;
}cylinder_t;

typedef struct{
  vec3 pos;
  f32  radius;
  u32  albedo;
}sphere_t;

// ----------------------------------------------------------------
typedef struct{
  vec3 pos, dir;
}ray_t;

typedef struct{
  u16 gtype;  // What type of object did we hit, and in which mesh?
  u32 id;     // The object ID, so that we know which object we hit!
  f32 t;      // The position of the hit in RAY COORDINATES. A ray is 1-dimensional, so its coordinates are 1-dimensional, too! Here we record *where* we hit the object!
}hit_t;

// ----------------------------------------------------------------------------------------------------------------------------#
#define BVH_NTHREADS          (1<<8)
#define BVH_STACK_NELEMS      16
#define BVH_MORTON_PRECISION  21

struct bvh_node_t{
  // Tree data!
  struct bvh_node_t* parent;
  struct bvh_node_t* left;
  struct bvh_node_t* right;
  int                is_leaf;

  // Geometric data!
  u32    primitive_idx;  // At each leaf, with the PRIMITIVE index we'll get the PRIMITIVE itself! This is a bit faster than keeping the full PRIMITIVE data at each BVH node!
  aabb_t node_aabb;      // At each NONLEAF we meet, we use ray/AABB intersection! At each LEAF we meet, we use ray/PRIMITIVE intersection!

  // Optimization data! 
  // u32 triangle_min, triangle_max;  // Index of the min and max PRIMITIVE!
  // f32 cost, area;
};

// ----------------------------------------------------------------
typedef struct{
  u16   gtype;  // What type of object did we hit, and in which mesh?
  i32   nelems;
  void* data;

  // BVH stuff
  aabb_t* aabbs;    // Bouding boxes for each node?
  u64*    mortons;  // Morton codes, ie. the positions in under the Z-curve ordering!
  u32*    idxs;

  struct bvh_node_t* tree_nodes;
  struct bvh_node_t* tree_leaves;
  i32*               tree_semaphore;  // This'll come in handy during BVH contruction and optimization!

  // Geometric transformations!
  vec3 mov, rot;  // @rot is a vector where each component is the rotation angle along an axis!
}mesh_t;

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  types: high-level! */
#include <cuda.h>  // CUDA Driver API!

typedef struct{
  // Stuff here DOESN'T need a CUDA context active!
  int        id;
  CUdevice   dev;
  CUcontext  ctx;

  // Stuff here DOES need a CUDA context active!
  CUstream   stream1;

  CUmodule   mod_bvh;
  CUfunction ker_mesh_radixtree_make, ker_mesh_aabbtree_make, ker_sort_idxs_by_mortons;

  CUmodule   mod_pt;
  CUfunction ker_light_shader, ker_mesh0_shader, ker_pixel_shader;
}cudev_t;

typedef struct{
  i32 frame;
  i32 nsamples, nbounces;
  u32 seed;

  i64   img_dim_h, img_dim_w;
  u32*  img_data;  // App-global frame-buffer! An array of u32's!  NOTE! NOT owned by this struct!
  i64   tile_pos_r, tile_pos_c;
  i64   tile_dim_h, tile_dim_w;
  u32*  tile_data;   // GPU-local      frame-buffer! An array of u32' s!  NOTE! Owned by this struct!
  vec3* tile_accum;  // GPU-local accumulate-buffer! An array of vec3's!  NOTE! Owned by this struct!

  // Camera geometric stuff!
  vec3 cam_pos, cam_dir;
  vec3 cam_mov, cam_rot;  // Camera geometric transformations!
}fb_t;

typedef struct{  // Data structure to handle the geometry in the scene!
  i32 nelems;  // Total number of geometric primitives in the scene!

  i32         nlights;
  i32         ntriangles;
  i32         ncylinders;
  i32         nspheres;
  light_t*    lights;
  triangle_t* triangles;
  cylinder_t* cylinders;
  sphere_t*   spheres;

  mesh_t mesh0;  // mesh of lights!
  mesh_t mesh1;  // triangle mesh!
  mesh_t mesh2;  // cylinder mesh!
  mesh_t mesh3;  // sphere mesh!

  // Geometric transformations!
  vec3 mov, rot;  // @rot is a vector where each component is the rotation angle along an axis!
}scene_t;

// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
typedef struct{
  cudev_t dev;
  fb_t    fb;
  scene_t scene;
}gpu_t;
