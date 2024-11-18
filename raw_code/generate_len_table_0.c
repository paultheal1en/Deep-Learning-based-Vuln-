static void generate_len_table(uint8_t *dst, uint64_t *stats, int size){

    heap_elem_t h[size];

    int up[2*size];

    int len[2*size];

    int offset, i, next;



    for(offset=1; ; offset<<=1){

        for(i=0; i<size; i++){

            h[i].name = i;

            h[i].val = (stats[i] << 8) + offset;

        }

        for(i=size/2-1; i>=0; i--)

            heap_sift(h, i, size);



        for(next=size; next<size*2-1; next++){

            // merge the two smallest entries, and put it back in the heap

            uint64_t min1v = h[0].val;

            up[h[0].name] = next;

            h[0].val = INT64_MAX;

            heap_sift(h, 0, size);

            up[h[0].name] = next;

            h[0].name = next;

            h[0].val += min1v;

            heap_sift(h, 0, size);

        }



        len[2*size-2] = 0;

        for(i=2*size-3; i>=size; i--)

            len[i] = len[up[i]] + 1;

        for(i=0; i<size; i++) {

            dst[i] = len[up[i]] + 1;

            if(dst[i] > 32) break;

        }

        if(i==size) break;

    }

}
