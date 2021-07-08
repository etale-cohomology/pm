#include "pm.h"


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
static __device__ __forceinline__ int gpu_longest_common_prefix(int i, int j, int len){  // Longest common prefix for morton code!
  if(0<=j && j<len) return __clz(i^j);
  else              return -1;
}

static __device__ __forceinline__ void gpu_aabb_merge(aabb_t b1, aabb_t b2, aabb_t* b3){
  b3->min.x0 = fminf(b1.min.x0,b2.min.x0);  b3->max.x0 = fmaxf(b1.max.x0,b2.max.x0);
  b3->min.x1 = fminf(b1.min.x1,b2.min.x1);  b3->max.x1 = fmaxf(b1.max.x1,b2.max.x1);
  b3->min.x2 = fminf(b1.min.x2,b2.min.x2);  b3->max.x2 = fmaxf(b1.max.x2,b2.max.x2);
  return;
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section */
extern "C" __global__ void ker_mesh_radixtree_make(int radixtree_nnodes, bvh_node_t* radixtree_nodes, bvh_node_t* radixtree_leaves){  // Radix tree construction kernel. Algorithm described in karras2012 paper. Node-wise parallel!
  int thr_idx = blockIdx.x*blockDim.x + threadIdx.x;  if(thr_idx>=radixtree_nnodes) return;
  int d       = gpu_longest_common_prefix(thr_idx, thr_idx+1, radixtree_nnodes+1) - gpu_longest_common_prefix(thr_idx, thr_idx-1, radixtree_nnodes+1)>0 ? 1 : -1;  // Run radix tree construction algorithm! Determine direction of the range (+1 or -1)
  int sigMin  = gpu_longest_common_prefix(thr_idx, thr_idx-d, radixtree_nnodes+1);  // Compute upper bound for the length of the range!
  int lmax    = 2;
  while(gpu_longest_common_prefix(thr_idx, thr_idx + lmax*d, radixtree_nnodes+1) > sigMin)
    lmax *= 2;

  int l       = 0;  // Find the other end using binary search!
  int divider = 2;
  for(int t = lmax / divider; t>=1; divider*=2){
    if(gpu_longest_common_prefix(thr_idx, thr_idx + (l + t) * d, radixtree_nnodes+1) > sigMin)
      l += t;
    t = lmax/divider;
  }

  int j       = thr_idx + l * d;  //printf("thr_idx:%d d:%d lmax:%d l:%d j:%d \n",thr_idx , d, lmax, l, j);
  int sigNode = gpu_longest_common_prefix(thr_idx, j, radixtree_nnodes+1);  // Find the split position using binary search
  int s       = 0;
  divider     = 2;
  for(int t = (l+(divider-1)) / divider; t>=1; divider*=2){
    if(gpu_longest_common_prefix(thr_idx, thr_idx + (s+t)*d, radixtree_nnodes+1) > sigNode)
      s += t;
      t = (l+(divider-1)) / divider;  // BUG? Should this go inside the `if`, or outside???
  }

  int         gamma   = thr_idx + s*d + min(d,0);
  bvh_node_t* current = radixtree_nodes + thr_idx;  // Output child pointers
  if(min(thr_idx,j)==gamma+0){ current->left  = radixtree_leaves+gamma+0;  (radixtree_leaves+gamma+0)->parent = current; }
  else{                        current->left  = radixtree_nodes +gamma+0;  (radixtree_nodes +gamma+0)->parent = current; }
  if(max(thr_idx,j)==gamma+1){ current->right = radixtree_leaves+gamma+1;  (radixtree_leaves+gamma+1)->parent = current; }
  else{                        current->right = radixtree_nodes +gamma+1;  (radixtree_nodes +gamma+1)->parent = current; }

  // current->triangle_min = min(thr_idx,j);
  // current->triangle_max = max(thr_idx,j);
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section BVH Construction kernel! Algorithm described in karras2012 paper (bottom-up approach)! */
extern "C" __global__ void ker_mesh_aabbtree_make(u32 nprimitives, aabb_t* aabbs,u32* idxs_sorted, bvh_node_t* bvh_nodes,bvh_node_t* bvh_leaves, i32* bvh_semaphore){
  u32         thr_idx       = blockIdx.x*blockDim.x + threadIdx.x;  if(thr_idx>=nprimitives) return;
  bvh_node_t* leaf          = bvh_leaves + thr_idx;
  u32         primitive_idx = idxs_sorted[thr_idx];  // Handle leaf first! Retrieve the sorted primitives, which have been sorted by Morton code!

  leaf->primitive_idx = primitive_idx;  // leaf->triangle = primitives[primitive_idx];  // Storing the triangle full data is just a bit slower than storing just the triangle index!
  leaf->node_aabb     = aabbs[primitive_idx];  // Out of the primitive AABBs, we'll build the AABB for the whole node!

  bvh_node_t* current     = leaf->parent;
  i32         current_idx = current - bvh_nodes;
  i32         result      = atomicAdd(bvh_semaphore+current_idx, 1);

  for(;;){  // Go up and handle internal nodes!
    if(result==0) return;
    gpu_aabb_merge(current->left->node_aabb, current->right->node_aabb, &(current->node_aabb));  if(current==bvh_nodes) return;  // If current is root, return!
    current     = current->parent;
    current_idx = current - bvh_nodes;
    result      = atomicAdd(bvh_semaphore+current_idx, 1);
  }
}


// ----------------------------------------------------------------------------------------------------------------------------#
// ----------------------------------------------------------------------------------------------------------------------------#
/* @section Sort indices by morton codes! */
extern "C" __global__ void ker_sort_idxs_by_mortons(u32 nprimitives, u64* mortons, u32* idxs){
}
