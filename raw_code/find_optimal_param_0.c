static int find_optimal_param(uint32_t sum, int n)

{

    int k, k_opt;

    uint32_t nbits[MAX_RICE_PARAM+1];



    k_opt = 0;

    nbits[0] = UINT32_MAX;

    for(k=0; k<=MAX_RICE_PARAM; k++) {

        nbits[k] = rice_encode_count(sum, n, k);

        if(nbits[k] < nbits[k_opt]) {

            k_opt = k;

        }

    }

    return k_opt;

}
