static void create_map(vorbis_context *vc, unsigned floor_number)

{

    vorbis_floor *floors = vc->floors;

    vorbis_floor0 *vf;

    int idx;

    int blockflag, n;

    int32_t *map;



    for (blockflag = 0; blockflag < 2; ++blockflag) {

        n = vc->blocksize[blockflag] / 2;

        floors[floor_number].data.t0.map[blockflag] =

            av_malloc((n + 1) * sizeof(int32_t)); // n + sentinel



        map =  floors[floor_number].data.t0.map[blockflag];

        vf  = &floors[floor_number].data.t0;



        for (idx = 0; idx < n; ++idx) {

            map[idx] = floor(BARK((vf->rate * idx) / (2.0f * n)) *

                             (vf->bark_map_size / BARK(vf->rate / 2.0f)));

            if (vf->bark_map_size-1 < map[idx])

                map[idx] = vf->bark_map_size - 1;

        }

        map[n] = -1;

        vf->map_size[blockflag] = n;

    }



    for (idx = 0; idx <= n; ++idx) {

        av_dlog(NULL, "floor0 map: map at pos %d is %d\n", idx, map[idx]);

    }

}
