#include "pm.h"


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
static __forceinline__ __device__ __host__ vec3 operator+(vec3  v, f32   s){  return {v.x0+s, v.x1+s, v.x2+s};  }                 // Scalar addition!
static __forceinline__ __device__ __host__ vec3 operator-(vec3  v, f32   s){  return {v.x0-s, v.x1-s, v.x2-s};  }                 // Scalar subtraction!
static __forceinline__ __device__ __host__ vec3 operator*(f32   s, vec3  v){  return {s*v.x0, s*v.x1, s*v.x2};  }                 // Scalar multiplication!
static __forceinline__ __device__ __host__ vec3 operator/(vec3  v, f32   s){  return {v.x0/s, v.x1/s, v.x2/s};  }                 // Scalar division!
static __forceinline__ __device__ __host__ vec3 operator+(vec3 v0, vec3 v1){  return {v0.x0+v1.x0, v0.x1+v1.x1, v0.x2+v1.x2};  }  // Vector elementwise addition!
static __forceinline__ __device__ __host__ vec3 operator-(vec3 v0, vec3 v1){  return {v0.x0-v1.x0, v0.x1-v1.x1, v0.x2-v1.x2};  }  // Vector elementwise subtraction!
static __forceinline__ __device__ __host__ vec3 operator*(vec3 v0, vec3 v1){  return {v0.x0*v1.x0, v0.x1*v1.x1, v0.x2*v1.x2};  }  // Vector elementwise multiplication!
static __forceinline__ __device__ __host__ vec3 operator/(vec3 v0, vec3 v1){  return {v0.x0/v1.x0, v0.x1/v1.x1, v0.x2/v1.x2};  }  // Vector elementwise division!

static __forceinline__ __device__ __host__ f32 dot(  vec3 v0, vec3 v1){  return v0.x0*v1.x0 + v0.x1*v1.x1 + v0.x2*v1.x2;  }  // Quite important for triangle intersection and a bit for the path tracer!
static __forceinline__ __device__ __host__ vec3 cross(vec3 v0, vec3 v1){  // Homology of R3: 0 --> 1 --> 2 --> 0 --> 1 --> 2 --> 0 --> ...
  return {v0.x1*v1.x2 - v0.x2*v1.x1,   // 0 --> 1 --> 2
          v0.x2*v1.x0 - v0.x0*v1.x2,   // 1 --> 2 --> 0
          v0.x0*v1.x1 - v0.x1*v1.x0};  // 2 --> 0 --> 1
}
static __forceinline__ __device__ __host__ vec3 normalize(vec3 v){  f32 s = rsqrtf(dot(v,v));  return {s*v.x0, s*v.x1, s*v.x2};  }

// ----------------------------------------------------------------------------------------------------------------------------#
static __forceinline__ __device__ __host__ quat operator*(f32   s, quat  q){  return {s*q.x0, s*q.x1, s*q.x2, s*q.x3};  }  // Scalar multiplication!
static __forceinline__ __device__ __host__ quat operator+(quat v0, quat v1){  return {v0.x0+v1.x0, v0.x1+v1.x1, v0.x2+v1.x2, v0.x3+v1.x3};  }  // Vector addition!

static __forceinline__ __device__ __host__ quat operator*(quat q0, quat q1){
  return {q0.x0*q1.x0 - q0.x1*q1.x1 - q0.x2*q1.x2 - q0.x3*q1.x3,
          q0.x0*q1.x1 + q0.x1*q1.x0 + q0.x2*q1.x3 - q0.x3*q1.x2,
          q0.x0*q1.x2 - q0.x1*q1.x3 + q0.x2*q1.x0 + q0.x3*q1.x1,
          q0.x0*q1.x3 + q0.x1*q1.x2 - q0.x2*q1.x1 + q0.x3*q1.x0};
}
static __forceinline__ __device__ __host__ quat conj(     quat q){            return {q.x0, -q.x1, -q.x2, -q.x3};  }  // The quaternion inverse of a quaternion `q` is just `conj(q) / quad(q)`, just like for complex numbers!
static __forceinline__ __device__ __host__ f32  dot(      quat q0, quat q1){  return q0.x0*q1.x0 + q0.x1*q1.x1 + q0.x2*q1.x2 + q0.x3*q1.x3;  }  // Quite important for triangle intersection and a bit for the path tracer!
static __forceinline__ __device__ __host__ quat normalize(quat q){            return rsqrtf(dot(q,q)) * q;  }

static __forceinline__ __device__ __host__ quat versor(f32 angle, vec3 dir){
  vec3 v = __sinf(.5*angle)*normalize(dir);
  return {__cosf(.5*angle), v.x0,v.x1,v.x2};  // If @dir isn't a `direction vector` (ie. a unit vector), then the rotation speed is not constant, methinks!
}
static __forceinline__ __device__ __host__ vec3 qrotl(vec3 v, quat versor){  // WARN! @versor must be a unit-quaternion!
  quat p_rot = versor * (quat){0,v.x0,v.x1,v.x2} * conj(versor);  // Left-conjugation by @versor! The quaternion-inverse of a unit-quaternion is its quaternion-conjugate!
  return {p_rot.x1, p_rot.x2, p_rot.x3};
}
static __forceinline__ __device__ __host__ vec3 qrotr(vec3 v, quat versor){  // WARN! @versor must be a unit-quaternion!
  quat p_rot = conj(versor) * (quat){0,v.x0,v.x1,v.x2} * versor;  // Right-conjugation by @versor! The quaternion-inverse of a unit-quaternion is its quaternion-conjugate!
  return {p_rot.x1, p_rot.x2, p_rot.x3};
}

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ vec3 clamp01(vec3 v){  return {__saturatef(v.x0), __saturatef(v.x1), __saturatef(v.x2)};  }

__forceinline__ __device__ f32  fminf(f32 a, f32 b, f32 c){       return fminf(fminf(a,b),c); }
__forceinline__ __device__ f32  fmaxf(f32 a, f32 b, f32 c){       return fmaxf(fmaxf(a,b),c); }
__forceinline__ __device__ f32  fminf(vec3 a){                    return fminf(a.x0,a.x1,a.x2); }
__forceinline__ __device__ f32  fmaxf(vec3 a){                    return fmaxf(a.x0,a.x1,a.x2); }
__forceinline__ __device__ vec3 fminf(vec3 a0, vec3 a1){          return {fminf(a0.x0,a1.x0), fminf(a0.x1,a1.x1), fminf(a0.x2,a1.x2)}; }
__forceinline__ __device__ vec3 fmaxf(vec3 a0, vec3 a1){          return {fmaxf(a0.x0,a1.x0), fmaxf(a0.x1,a1.x1), fmaxf(a0.x2,a1.x2)}; }
__forceinline__ __device__ vec3 fminf(vec3 a0, vec3 a1, vec3 a2){ return fminf(a2,fminf(a1,a0)); }
__forceinline__ __device__ vec3 fmaxf(vec3 a0, vec3 a1, vec3 a2){ return fmaxf(a2,fmaxf(a1,a0)); }

// ----------------------------------------------------------------------------------------------------------------------------#
__forceinline__ __device__ f32  rgb_gamma_decode(f32 channel){  return __powf(channel, 2.2/1);  }
__forceinline__ __device__ f32  rgb_gamma_encode(f32 channel){  return __powf(channel, 1/2.2);  }
__forceinline__ __device__ f32  rgb_u8_to_f32(   u32 channel){  return rgb_gamma_decode(channel/255.); }       // Read "from disk" "to memory", map from nonlinear color space (for monitor displaying) to linear color space (for computations)!
__forceinline__ __device__ u32  rgb_f32_to_u8(   f32 channel){  return 255.*rgb_gamma_encode(channel) + .5; }  // Write "from memory" "to disk", map from linear color space (for computations) to nonlinear color space (for monitor displaying)!
__forceinline__ __device__ vec3 bgr8u_to_rgb32f(u32 bgr8u){
  return {rgb_u8_to_f32((bgr8u>>0x10)&0xff),
          rgb_u8_to_f32((bgr8u>>0x08)&0xff),
          rgb_u8_to_f32((bgr8u>>0x00)&0xff)};
}
__forceinline__ __device__ vec3 rgb8u_to_rgb32f(u32 bgr8u){
  return {rgb_u8_to_f32((bgr8u>>0x00)&0xff),
          rgb_u8_to_f32((bgr8u>>0x08)&0xff),
          rgb_u8_to_f32((bgr8u>>0x10)&0xff)};
}
__forceinline__ __device__ u32 rgb32f_to_rgbu8(vec3 rgbf32){
  return (rgb_f32_to_u8(rgbf32.x0)<<0x00) |
         (rgb_f32_to_u8(rgbf32.x1)<<0x08) |
         (rgb_f32_to_u8(rgbf32.x2)<<0x10);
}

__forceinline__ __device__ f32 rand_f32(u32* seed0, u32* seed1){  // RNG from github.com/gz/rust-raytracer
  *seed0  = 36969*(*seed0&0xffff) + (*seed0>>0x10);
  *seed1  = 18489*(*seed1&0xffff) + (*seed1>>0x10);
  u32 val_u32 = 0x40000000 | (((*seed0<<0x10) + *seed1) & 0x007fffff);
  return .5f * (*(f32*)&val_u32) - 1.f;
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section  path tracer!  Based on Sam Lapere's path tracer: */

// ----------------------------------------------------------------------------------------------------------------------------#
static __forceinline__ __device__ f32 pt_aabb_intersect(aabb_t self, ray_t ray){  // Intersect this primitive with a ray! Return the distance, or 0 if there's no hit!
  vec3 t_min    = (self.min - ray.pos) / ray.dir;
  vec3 t_max    = (self.max - ray.pos) / ray.dir;
  vec3 real_min = fminf(t_min, t_max);
  vec3 real_max = fmaxf(t_min, t_max);
  f32  maxmin   = fmaxf(real_min);
  f32  minmax   = fminf(real_max);
  if(minmax>=maxmin)  return maxmin;
  else                return 0.f;
}

static __forceinline__ __device__ vec3 pt_aabb_normal(aabb_t self, vec3 p){
  vec3 normal;
  if(     fabsf(self.min.x0-p.x0)<PM_EPSILON) normal={-1, 0, 0};  // Only 6 possible normals, one possibility for each face of the box!
  else if(fabsf(self.max.x0-p.x0)<PM_EPSILON) normal={ 1, 0, 0};
  else if(fabsf(self.min.x1-p.x1)<PM_EPSILON) normal={ 0,-1, 0};
  else if(fabsf(self.max.x1-p.x1)<PM_EPSILON) normal={ 0, 1, 0};
  else if(fabsf(self.min.x2-p.x2)<PM_EPSILON) normal={ 0, 0,-1};
  else                                       normal={ 0, 0, 1};
  return normal;  // A normal MUST be a direction vector, ie. a unit vector! But notice each vector above IS already normalized! =D
}

// ----------------------------------------------------------------
static __forceinline__ __device__ f32 pt_triangle_intersect(triangle_t self, ray_t ray){  // Intersect this geometric primitive with a ray! Return the distance, or 0 if there's no hit!
  vec3 op      = ray.pos - self.vert0;
  vec3 pvec    = cross(ray.dir,self.edge02);
  f32  det     = __fdividef(1.f, dot(self.edge01,pvec));  // `det<0` means FRONT!
  f32  u       = det * dot(op,pvec);       if(u<0.f || u  >1.f) return 0.f;  // No intersection! Early exit DOES help!
  vec3 qvec    = cross(op,self.edge01);
  f32  v       = det * dot(ray.dir,qvec);  if(v<0.f || u+v>1.f) return 0.f;  // No intersection!
  return det * dot(self.edge02,qvec);  // `det<0` means FRONT!
}
static __forceinline__ __device__ vec3 pt_triangle_normal(triangle_t self, vec3 x){  // A triangle has to curvature, so it's normal vector field is a CONSTANT vector field: it's value is the same across all points on the surface!
  return normalize(cross(self.edge01,self.edge02));  // The cross product of two triangle edges yields a vector orthogonal to the triangle plane! A normal MUST be a direction vector, ie. a unit vector!
}

// ----------------------------------------------------------------
static __forceinline__ __device__ f32 pt_light_intersect(light_t self, ray_t ray){  // Intersect this geometric primitive with a ray! Return the distance, or 0 if there's no hit!
  return pt_triangle_intersect({self.vert0, self.edge01, self.edge02}, ray);
}
static __forceinline__ __device__ vec3 pt_light_normal(light_t self, vec3 x){  // A triangle has to curvature, so it's normal vector field is a CONSTANT vector field: it's value is the same across all points on the surface!
  return normalize(cross(self.edge01,self.edge02));  // The cross product of two triangle edges yields a vector orthogonal to the triangle plane! A normal MUST be a direction vector, ie. a unit vector!
}

// ----------------------------------------------------------------
static __forceinline__ __device__ f32 pt_cylinder_intersect(cylinder_t self, ray_t ray){  // Intersect this geometric primitive with a ray! Return the distance, or 0 if there's no hit!
  if(self.height==0.f) return 0.f;  // This allows us to have "trivial" primitives in the mesh and not break the path tracer!
  vec3 op         = ray.pos - self.pos;
  f32  op_dir     = dot(op,self.dir);
  vec3 op_dir_vec = op_dir*self.dir;
  vec3 x          = ray.dir - dot(ray.dir,self.dir)*self.dir;
  vec3 y          = op - op_dir_vec;

  f32 A = dot(x,x);
  f32 B = 2.f*dot(x,y);
  f32 C = dot(y,y) - self.radius*self.radius;

  f32 dscr = B*B - 4*A*C;  // Discriminant of the quadratic equation!
  if(dscr<0.f)  return 0.f;
  else{  f32 top;
    vec3 bot_base = self.pos;
    vec3 top_base = self.pos + self.height*self.dir;
    top = -B-sqrtf(dscr);  if(top>0.f){  f32 t=top/(2.f*A); vec3 q=ray.pos+t*ray.dir; if(dot(self.dir, q-bot_base)>0.f && dot(self.dir, q-top_base)<0.f) return t;  else return 0.f;  }  // No 'FAST intrinsic for sqrt?
    top = -B+sqrtf(dscr);  if(top>0.f){  f32 t=top/(2.f*A); vec3 q=ray.pos+t*ray.dir; if(dot(self.dir, q-bot_base)>0.f && dot(self.dir, q-top_base)<0.f) return t;  else return 0.f;  }  // No 'FAST intrinsic for sqrt?
  }
  return 0.f;
}
static __forceinline__ __device__ vec3 pt_cylinder_normal(cylinder_t self, vec3 x){  // Return the normal to the cylinder at a point @x
  vec3 a = x - self.pos;
  vec3 b = self.dir;
  vec3 r = a - dot(a,b)*b;  // a - dot(a,b)/dot(b,b)*b;  // Vector rejection of `a` on `b`, but optimized since `b` is a unit vector!
  return normalize(r);
}

// ----------------------------------------------------------------
static __forceinline__ __device__ f32 pt_sphere_intersect(sphere_t self, ray_t ray){
  if(self.radius==0.f) return 0.f;
  vec3 op   = self.pos - ray.pos;
  f32  b    = dot(op, ray.dir);  // `b` term in the sphere's quadratic equation
  f32  disc = b*b - dot(op,op) + self.radius*self.radius;  // The disc in the sphere's quadratic equation

  if(disc<0.f) return 0.f;  // If the discriminant is negative, then there's only complex roots!
  disc = __fsqrt_rn(disc);  // If discriminant non-negative, then check for roots using negative and positive discriminant!

  f32 t;
  t = b-disc;  if(t>0.f) return t;  // Pick closest point in front of ray origin?
  t = b+disc;  if(t>0.f) return t;
  return 0.f;
}
static __forceinline__ __device__ vec3 pt_sphere_normal(sphere_t self, vec3 x){
  return normalize(x - self.pos);
}

// ----------------------------------------------------------------
static __forceinline__ __device__ hit_t pt_scene_intersect(ray_t ray, scene_t scene){
  hit_t hit = {gtype:GTYPE_UNKNOWN, id:0xffffffff, t:1e38};

  // ----------------------------------------------------------------
  // Record the position of the closest intersection point in RAY COORDINATES (which are 1-dimensional, so you need a single number), and also the ID of the object in question
  for(int i=0; i<scene.nlights; ++i){
    f32 t     = pt_light_intersect(scene.lights[i], ray);  if(t<PM_EPSILON || t>hit.t) continue;
    hit.t     = t;
    hit.id    = i;
    hit.gtype = GTYPE_LIGHT;
  }
#if 0
  for(int i=0; i<scene.ntriangles; ++i){
    f32 t     = pt_triangle_intersect(scene.triangles[i], ray);  if(t<PM_EPSILON || t>hit.t) continue;
    hit.t     = t;
    hit.id    = i;
    hit.gtype = GTYPE_TRIANGLE;
  }
#endif
#if 0
  for(i32 i=0; i<scene.ncylinders; ++i){
    f32 t     = pt_cylinder_intersect(scene.cylinders[i], ray);  if(t<PM_EPSILON || t>hit.t) continue;
    hit.t     = t;
    hit.id    = i;
    hit.gtype = GTYPE_CYLINDER;
  }
#endif
#if 0
  for(i32 i=0; i<scene.nspheres; ++i){
    f32 t     = pt_sphere_intersect(scene.spheres[i], ray);  if(t<PM_EPSILON || t>hit.t) continue;
    hit.t     = t;
    hit.id    = i;
    hit.gtype = GTYPE_SPHERE;
  }
#endif

  // ----------------------------------------------------------------
#if 1  // Mesh intersection with a BVH! 5x-10x faster than without a BVH!
  {
    bvh_node_t* node_stack[BVH_STACK_NELEMS];  // Use static allocation! Use stack to traverse BVH to save space, cost is O(height)
    i32 stack_idx           = BVH_STACK_NELEMS;
    node_stack[--stack_idx] = scene.mesh0.tree_nodes;  // The root is the LAST element of the stack, ie. the element at position BVH_STACK_NELEMS!

    while(stack_idx != BVH_STACK_NELEMS){  // Stack-based recursion!
      bvh_node_t* node = node_stack[stack_idx++];
      if(!pt_aabb_intersect(node->node_aabb, ray)) continue;  // In BVH-based path tracing, MOST of the BVH intersection-traversal is ray/AABB! To intersect a NON-LEAF (every level other than the last), we use ray/AABB intersection! To intersect a LEAF (only the last level), we use ray/triangle intersection!

      if(!node->is_leaf){  // We'll only call the primitive-intersection routine at the LEAVES of the BVH!
        node_stack[--stack_idx] = node->right;
        node_stack[--stack_idx] = node->left;  // if(stack_idx<0){ printf("The BVH stack not big enough! Increase BVH_STACK_NELEMS!\n"); return hit; }
        continue;
      }

      // The spheres don't need an AABB! The non-leaf nodes don't need a sphere or a sphere index!
      u32      id     = node->primitive_idx;  // sphere_t primitive = node->primitive;  // Storing the sphere full data is just a bit slower than storing just the sphere index!
      sphere_t sphere = ((sphere_t*)scene.mesh0.data)[id];
      f32 t           = pt_sphere_intersect(sphere, ray);  if(t<PM_EPSILON || t>hit.t) continue;  // Ray/sphere intersection only happens at the leaves!
      hit.t           = t;
      hit.id          = id;  // NOW we need the sphere struct to hold the sphere index, since we're NOT traversing the sphere array in linear order!
      hit.gtype       = scene.mesh0.gtype;
    }
  }
#endif  // Mesh intersection with a BVH!

  // ----------------------------------------------------------------
  return hit;
}

// ----------------------------------------------------------------
static __forceinline__ __device__ vec3 pt_normal_out(vec3 normal, vec3 ray_dir){
  return dot(normal,ray_dir)<0 ? normal : -1*normal;  // "Outwards" normal, to create a "bounce"!
}
static __forceinline__ __device__ vec3 pt_hemisphere_randdir(vec3 normal, uint* seed_x, uint* seed_y){  // Sample a random direction on the dome/hemisphere around the hitpoint base on the normal at that point!
  // Compute local orthonormal basis uvw at hitpoint, to compute the (random) ray direction. 1st vector is the normal, 2nd vector is orthogonal to 1st, 3rd vector is orthogonal to first others
  vec3 basis_w = normal;
  vec3 axis    = fabs(basis_w.x0)<.1f ? (vec3){1,0,0} : (vec3){0,1,0};
  vec3 basis_u = normalize(cross(axis, basis_w));  // We shouldn't need to normalize this, but, if we don't, then we introduce artifacts!
  vec3 basis_v = cross(basis_w, basis_u);          // Right-handed uvw-basis! The homology is: u -> v -> w -> u -> ...

  // All our geometric primitives (just triangles) are diffuse, which reflect light uniformly in all directions! Generate random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)
  f32 rand_tau  = rand_f32(seed_x,seed_y) * M_TAU;  // Get random number on unit circle for azimuth
  f32 rand_one  = rand_f32(seed_x,seed_y);          // Get random number for elevation
  f32 rand_sqrt = sqrtf(rand_one);  // No FAST intrinsic for sqrt?

  f32 cos_tau, sin_tau; __sincosf(rand_tau, &sin_tau,&cos_tau);
  return cos_tau*rand_sqrt*basis_u + sin_tau*rand_sqrt*basis_v + sqrtf(1.f-rand_one)*basis_w;  // Random ray direction on the hemisphere/dome around a point! Cosine-weighted importance sampling, favours ray directions closer to normal direction!
}

// ----------------------------------------------------------------
static __device__ vec3 pt_radiance_path_integral(ray_t ray, fb_t fb, scene_t scene, uint* seed_x,uint* seed_y){  // i32 nlights,light_t* lights, i32 ntriangles,triangle_t* triangles, i32 ncylinders,cylinder_t* cylinders, i32 nspheres,sphere_t* spheres
  vec3 rgb  = {0,0,0};
  vec3 fade = {1,1,1};

  // 0) Scene intersection!
  for(int bounce=0; bounce<fb.nbounces; ++bounce){
    hit_t hit     = pt_scene_intersect(ray, scene);  if(hit.t==1e38f) return {0,0,0};  // No intersection/hit! Return black!
    vec3  hit_pos = ray.pos + hit.t*ray.dir;  // @hit_pos is the hit position in WORLD COORDINATES! @hit.t is the hit position in RAY COORDINATES!

    // ----------------------------------------------------------------
    vec3 obj_normal, obj_rgb, obj_emi;
    switch(hit.gtype){
      case GTYPE_LIGHT:{
      light_t light = scene.lights[hit.id];
      obj_normal    = pt_light_normal(light, hit_pos);
      obj_rgb       = {0,0,0};
      obj_emi       = light.emission;
      }break;
#if 0
      case GTYPE_TRIANGLE:{
      triangle_t triangle = scene.triangles[hit.id];
      obj_normal          = pt_triangle_normal(triangle, hit_pos);
      obj_rgb             = rgb8u_to_rgb32f(triangle.albedo);
      obj_emi             = {0,0,0};
      }break;
#endif
#if 0
      case GTYPE_CYLINDER:{
      cylinder_t cylinder = scene.cylinders[hit.id];
      obj_normal          = pt_cylinder_normal(cylinder, hit_pos);
      obj_rgb             = rgb8u_to_rgb32f(cylinder.albedo);
      obj_emi             = {0,0,0};
      }break;
#endif
#if 1
      case GTYPE_SPHERE:{
      sphere_t sphere = ((sphere_t*)scene.mesh0.data)[hit.id];
      obj_normal      = pt_sphere_normal(sphere, hit_pos);
      obj_rgb         = bgr8u_to_rgb32f(sphere.albedo);
      obj_emi         = {0,0,0};
      }break;
#endif
    }

    // ----------------------------------------------------------------
    vec3 obj_normal_out = pt_normal_out(obj_normal, ray.dir);  // "Outwards" normal, to create a "bounce"!
    vec3 bounce_dir     = pt_hemisphere_randdir(obj_normal, seed_x,seed_y);

    // 1) Light transport!
    rgb  = rgb + fade*obj_emi;                                // Add emission of current object to accumulated color (first term in rendering equation sum)
    fade = dot(obj_normal_out, bounce_dir) * obj_rgb * fade;  // Integrate/sum/accumulate the fade! Weigh light/color energy using cosine of angle between normal and incident light!

    // 2) Ray/path bouncing!
    ray.pos = hit_pos + 0.0001f*obj_normal_out;  // Launch a new raw starting by "bouncing" it from the object! Offset ray position slightly to prevent self intersection
    ray.dir = bounce_dir;
  }

  return rgb;
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
static __forceinline__ __device__ aabb_t aabb3d_sphere(sphere_t sphere){
  return {.min=sphere.pos - sphere.radius, .max=sphere.pos + sphere.radius};
}

static __forceinline__ __device__ u64 gpu_zorder3D(vec3 x, aabb_t mesh_aabb){  // Compute the 1D position of a 3D position @x in a 1D Z-order curve living in 3D space, given a particular (global) AABB! NOTE: The AABB must be GLOBAL for the whole mesh that the 3D position belongs to!
  x.x0         = (x.x0 - mesh_aabb.min.x0) / (mesh_aabb.max.x0 - mesh_aabb.min.x0);  // Map @x in @mesh_aabb to the 3D interval [0 .. 1]^3
  x.x1         = (x.x1 - mesh_aabb.min.x1) / (mesh_aabb.max.x1 - mesh_aabb.min.x1);  // Map @x in @mesh_aabb to the 3D interval [0 .. 1]^3
  x.x2         = (x.x2 - mesh_aabb.min.x2) / (mesh_aabb.max.x2 - mesh_aabb.min.x2);  // Map @x in @mesh_aabb to the 3D interval [0 .. 1]^3
  u64 morton_a = (u64)x.x0 * (1ull<<BVH_MORTON_PRECISION);                           // Map @x in @mesh_aabb to the 3D interval [0 .. 2**BVH_MORTON_PRECISION]^3, meaning each coordinate can be represented using BVH_MORTON_PRECISION bits! (Although I think we lose the highest in the mesh AABB?)
  u64 morton_b = (u64)x.x1 * (1ull<<BVH_MORTON_PRECISION);                           // Map @x in @mesh_aabb to the 3D interval [0 .. 2**BVH_MORTON_PRECISION]^3, meaning each coordinate can be represented using BVH_MORTON_PRECISION bits! (Although I think we lose the highest in the mesh AABB?)
  u64 morton_c = (u64)x.x2 * (1ull<<BVH_MORTON_PRECISION);                           // Map @x in @mesh_aabb to the 3D interval [0 .. 2**BVH_MORTON_PRECISION]^3, meaning each coordinate can be represented using BVH_MORTON_PRECISION bits! (Although I think we lose the highest in the mesh AABB?)

  u64 morton_code = 0x0000000000000000ull;
  for(int i=0; i<BVH_MORTON_PRECISION; ++i){  // Combine into 63 bits morton code!
    morton_code |=
    (((((morton_a >> (BVH_MORTON_PRECISION-1-i))) & 1) << ((BVH_MORTON_PRECISION-i)*3 - 1)) |
     ((((morton_b >> (BVH_MORTON_PRECISION-1-i))) & 1) << ((BVH_MORTON_PRECISION-i)*3 - 2)) |
     ((((morton_c >> (BVH_MORTON_PRECISION-1-i))) & 1) << ((BVH_MORTON_PRECISION-i)*3 - 3)));
  }
  return morton_code;
}

// ----------------------------------------------------------------------------------------------------------------------------#
extern "C" __global__ void ker_light_shader(fb_t fb, scene_t scene){  // Mesh of lights!
  f32 p = 1e1f;
  f32 x = 1e4f;
  f32 z = 1e2f;

  scene.lights[0] = {{-p,-p,+z}, { 0, 0,-x}, { 0,+x, 0}, {1.4,1.4,1.8}};              // Left face!
  scene.lights[1] = {{+p,-p,+z}, { 0,+x, 0}, { 0, 0,-x}, {1.4,1.4,1.8}};              // Right face!
  scene.lights[2] = {{-p,-p,+z}, {+x, 0, 0}, { 0, 0,-x}, {1.4,1.4,1.8}};              // Bottom face!
  scene.lights[3] = {{-p,+p,+z}, { 0, 0,-x}, {+x, 0, 0}, {1.4,1.4,1.8}};              // Top face!
  scene.lights[4] = {{-p,-p,-p}, {+x, 0, 0}, { 0,+x, 0}, rgb8u_to_rgb32f(0x080808)};  // Back face!
  scene.lights[5] = {{-p,-p,+z}, {+x, 0, 0}, { 0,+x, 0}, {1.4,1.4,1.8}};              // Front face!
}

// ----------------------------------------------------------------------------------------------------------------------------#
extern "C" __global__ void ker_mesh0_shader(fb_t fb, scene_t scene){
  i32       thr_lidx = blockIdx.x*blockDim.x + threadIdx.x;  if(thr_lidx>=scene.mesh0.nelems) return;
  sphere_t* spheres  = (sphere_t*)scene.mesh0.data;

  // ----------------------------------------------------------------
  quat rot_yz = versor(scene.rot.x0, {1,0,0});
  quat rot_zx = versor(scene.rot.x1, {0,1,0});
  quat rot_xy = versor(scene.rot.x2, {0,0,1});
  quat rot    = rot_yz*rot_zx*rot_xy;

  spheres[thr_lidx].pos = qrotl(spheres[thr_lidx].pos, rot) + scene.mov;

  // ----------------------------------------------------------------
  aabb_t mesh_aabb = {.min={-1,-1,-1}, .max={1,1,1}};  // Global AABB for ALL the triangles in this mesh!

  sphere_t sphere               = spheres[thr_lidx];
  scene.mesh0.aabbs[  thr_lidx] = aabb3d_sphere(sphere);
  scene.mesh0.mortons[thr_lidx] = gpu_zorder3D(sphere.pos, mesh_aabb);  // TODO! Everything is zero!
  scene.mesh0.idxs[   thr_lidx] = thr_lidx;  // printf("%d\n", scene.mesh0.idxs[thr_lidx]);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
extern "C" __global__ void ker_pixel_shader(fb_t fb, scene_t scene){
  i32  thr_lvl1_2idx_x = blockIdx.x*blockDim.x + threadIdx.x;
  i32  thr_lvl1_2idx_y = blockIdx.y*blockDim.y + threadIdx.y;  if(fb.tile_pos_c+fb.tile_dim_w<=fb.tile_pos_c+thr_lvl1_2idx_x || fb.tile_pos_r+fb.tile_dim_h<=fb.tile_pos_r+thr_lvl1_2idx_y) return;
  u32  seed_x          = thr_lvl1_2idx_x + fb.seed;
  u32  seed_y          = thr_lvl1_2idx_y + fb.seed;

  // TODO! Maybe we should hoist this out of the kernel, since the camera computations are the same for all threads and all GPUs!
  vec3 cam_pos    = fb.cam_pos + fb.cam_mov;
  quat cam_rot_yz = versor(fb.cam_rot.x0, {1,0,0});
  quat cam_rot_zx = versor(fb.cam_rot.x1, {0,1,0});
  quat cam_rot_xy = versor(fb.cam_rot.x2, {0,0,1});
  quat cam_rot    = cam_rot_yz*cam_rot_zx*cam_rot_xy;
  vec3 cam_dir    = qrotl(fb.cam_dir, cam_rot);
  vec3 cam_dir_x  = qrotl(.5f*PM_CAM_FOV * (vec3){(f32)fb.img_dim_w/fb.img_dim_h, 0, 0}, cam_rot);  // Cam ray is directed at the lower-left corner of the screen!
  vec3 cam_dir_y  =       .5f*PM_CAM_FOV * normalize(cross(cam_dir, -1*cam_dir_x));                 // Cam ray is directed at the lower-left corner of the screen!

  // ----------------------------------------------------------------
  vec3 px_rgb = {0,0,0};  // Final pixel color! Init to zero for each pixel!
  for(int sample=0; sample<fb.nsamples; ++sample){  // Samples per pixel! Camera rays are pushed forward to start in interior
    f32   cam_dx = (thr_lvl1_2idx_x + rand_f32(&seed_x,&seed_y)) / fb.img_dim_w - .5f;
    f32   cam_dy = (thr_lvl1_2idx_y + rand_f32(&seed_x,&seed_y)) / fb.img_dim_h - .5f + (f32)fb.tile_pos_r/fb.img_dim_h;
    vec3  px_dir = cam_dir + cam_dx*cam_dir_x + cam_dy*cam_dir_y;
    vec3  px_pos = cam_pos;
    ray_t px_ray = {px_pos, normalize(px_dir)};
    px_rgb       = px_rgb + 1.f/fb.nsamples * pt_radiance_path_integral(px_ray, fb, scene, &seed_x,&seed_y);
  }

  // ----------------------------------------------------------------
  u32 tile_lidx            = thr_lvl1_2idx_y*fb.img_dim_w + thr_lvl1_2idx_x;
  fb.tile_accum[tile_lidx] = fb.tile_accum[tile_lidx] + px_rgb;
  vec3 rgb                 = fb.tile_accum[tile_lidx] / (fb.frame+1);
  fb.tile_data[tile_lidx]  = rgb32f_to_rgbu8(clamp01(rgb));
}
