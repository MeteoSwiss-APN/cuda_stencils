#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "../defs.h"
#include "libjson.h"
#include "mesh.hpp"

#include "stencil_kernels.h"

int parse_uint(const char *str, unsigned int *output) {
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int main(int argc, char **argv) {

  unsigned int isize = 512;
  unsigned int jsize = 512;
  unsigned int ksize = 60;

  for (int i = 1; i < argc; i++) {
    if (!std::string("--isize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &isize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    if (!std::string("--jsize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &jsize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    if (!std::string("--ksize").compare(argv[i])) {
      if (++i >= argc || !parse_uint(argv[i], &ksize)) {
        std::cerr << "Wrong parsing" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  printf("Configuration :\n");
  printf("  isize: %d\n", isize);
  printf("  jsize: %d\n", jsize);
  printf("  ksize: %d\n", ksize);

  const size_t tot_size = isize * jsize * ksize;
  const size_t tsteps = 100;
  const size_t warmup_step = 10;

  mesh mesh_(isize, jsize, 2);
  mesh_.print();
  mesh_.test();

  std::ofstream fs("./perf_results.json", std::ios_base::app);
  JSONNode globalNode;
  globalNode.cast(JSON_ARRAY);
  globalNode.set_name("metrics");

  JSONNode dsize;
  dsize.set_name("domain");

  dsize.push_back(JSONNode("x", isize));
  dsize.push_back(JSONNode("y", jsize));
  dsize.push_back(JSONNode("z", ksize));


  printf("======================== FLOAT =======================\n");
  std::vector<double> timings(num_bench_st);
  std::fill(timings.begin(), timings.end(), 0);


  launch<float>(timings, mesh_, isize, jsize, ksize, tsteps, warmup_step);

  JSONNode precf;
  precf.set_name("float");

  JSONNode ftex;
  ftex.set_name("tex");

  printf("--------------------------\n");
  printf("copy : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[ucopy_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[ucopy_st]);

  printf("--------------------------\n");
  printf("copy mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[ucopymesh_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[ucopymesh_st]);

  printf("--------------------------\n");
  printf("on_cells : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[uoncells_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncells_st]);

  printf("--------------------------\n");
  printf("on_cells_ilp : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[uoncells_ilp_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncells_ilp_st]);


  printf("--------------------------\n");
  printf("on_cells_mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[uoncellsmesh_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncellsmesh_st]);
  printf("--------------------------\n");
  printf("on_cells_mesh_hilber : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
             (timings[uoncellsmesh_hilbert_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncellsmesh_hilbert_st]);

  printf("--------------------------\n");
  // w(b,d) r(a,b,c,fac1,fac2)
  printf("complex_on_cells : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(float) /
             (timings[complex_uoncells_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncells_st]);

  printf("--------------------------\n");
  printf("complex_on_cells_mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(float) /
             (timings[complex_uoncellsmesh_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncellsmesh_st]);
  printf("--------------------------\n");
  printf("complex_on_cells_mesh_hilber : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(float) /
             (timings[complex_uoncellsmesh_hilbert_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncellsmesh_hilbert_st]);

  {
    JSONNode stencils;
    stencils.set_name("stencils");
    stencils.push_back(JSONNode(
        "copy", tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
                    (timings[ucopy_st] / (double)(tsteps - (warmup_step + 1))) /
                    (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "copymesh",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
            (timings[ucopymesh_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
            (timings[uoncells_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_ilp",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
            (timings[uoncells_ilp_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_mesh",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(float) /
            (timings[uoncellsmesh_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_mesh_hilbert",
        tot_size * 2 * sizeof(double) /
            (timings[uoncellsmesh_hilbert_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));

    ftex.push_back(stencils);
    precf.push_back(ftex);
  }

  dsize.push_back(precf);


  printf("======================== DOUBLE =======================\n");
  launch<double>(timings, mesh_, isize, jsize, ksize, tsteps, warmup_step);

  JSONNode precd;
  precd.set_name("double");

  JSONNode dtex;
  dtex.set_name("tex");

  printf("--------------------------\n");
  printf("copy : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[ucopy_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[ucopy_st]);

  printf("--------------------------\n");
  printf("copy mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[ucopymesh_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[ucopymesh_st]);

  printf("--------------------------\n");
  printf("on_cells : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[uoncells_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncells_st]);

  printf("--------------------------\n");
  printf("on_cells_ilp : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[uoncells_ilp_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncells_ilp_st]);


  printf("--------------------------\n");
  printf("on_cells_mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[uoncellsmesh_st] / (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncellsmesh_st]);
  printf("--------------------------\n");
  printf("on_cells_mesh_hilber : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
             (timings[uoncellsmesh_hilbert_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[uoncellsmesh_hilbert_st]);

  printf("--------------------------\n");
  // w(b,d) r(a,b,c,fac1,fac2)
  printf("complex_on_cells : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(double) /
             (timings[complex_uoncells_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncells_st]);

  printf("--------------------------\n");
  printf("complex_on_cells_mesh : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(double) /
             (timings[complex_uoncellsmesh_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncellsmesh_st]);
  printf("--------------------------\n");
  printf("complex_on_cells_mesh_hilber : %f GB/s, time : %f \n",
         tot_size * 2 /*color*/ * 7 /* r/w */ * sizeof(double) /
             (timings[complex_uoncellsmesh_hilbert_st] /
              (double)(tsteps - (warmup_step + 1))) /
             (1024. * 1024. * 1024.),
         timings[complex_uoncellsmesh_hilbert_st]);

  {
    JSONNode stencils;
    stencils.set_name("stencils");
    stencils.push_back(JSONNode(
        "copy", tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
                    (timings[ucopy_st] / (double)(tsteps - (warmup_step + 1))) /
                    (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "copymesh",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
            (timings[ucopymesh_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
            (timings[uoncells_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_ilp",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
            (timings[uoncells_ilp_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_mesh",
        tot_size * 2 /*color*/ * 2 /* r/w */ * sizeof(double) /
            (timings[uoncellsmesh_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));
    stencils.push_back(JSONNode(
        "oncells_mesh_hilbert",
        tot_size * 2 * sizeof(double) /
            (timings[uoncellsmesh_hilbert_st] / (double)(tsteps - (warmup_step + 1))) /
            (1024. * 1024. * 1024.)));

    dtex.push_back(stencils);
    precd.push_back(dtex);
  }

  dsize.push_back(precd);

  globalNode.push_back(dsize);

  fs << globalNode.write_formatted() << std::endl;

}
