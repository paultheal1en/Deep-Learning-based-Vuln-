void ff_init_elbg(int *points, int dim, int numpoints, int *codebook,

                  int numCB, int max_steps, int *closest_cb,

                  AVLFG *rand_state)

{

    int i, k;



    if (numpoints > 24*numCB) {

        /* ELBG is very costly for a big number of points. So if we have a lot

           of them, get a good initial codebook to save on iterations       */

        int *temp_points = av_malloc(dim*(numpoints/8)*sizeof(int));

        for (i=0; i<numpoints/8; i++) {

            k = (i*BIG_PRIME) % numpoints;

            memcpy(temp_points + i*dim, points + k*dim, dim*sizeof(int));

        }



        ff_init_elbg(temp_points, dim, numpoints/8, codebook, numCB, 2*max_steps, closest_cb, rand_state);

        ff_do_elbg(temp_points, dim, numpoints/8, codebook, numCB, 2*max_steps, closest_cb, rand_state);



        av_free(temp_points);



    } else  // If not, initialize the codebook with random positions

        for (i=0; i < numCB; i++)

            memcpy(codebook + i*dim, points + ((i*BIG_PRIME)%numpoints)*dim,

                   dim*sizeof(int));



}
