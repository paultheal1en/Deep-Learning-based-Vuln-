static void fill_mbaff_ref_list(H264Context *h){

    int list, i, j;

    for(list=0; list<2; list++){

        for(i=0; i<h->ref_count[list]; i++){

            Picture *frame = &h->ref_list[list][i];

            Picture *field = &h->ref_list[list][16+2*i];

            field[0] = *frame;

            for(j=0; j<3; j++)

                field[0].linesize[j] <<= 1;

            field[1] = field[0];

            for(j=0; j<3; j++)

                field[1].data[j] += frame->linesize[j];



            h->luma_weight[list][16+2*i] = h->luma_weight[list][16+2*i+1] = h->luma_weight[list][i];

            h->luma_offset[list][16+2*i] = h->luma_offset[list][16+2*i+1] = h->luma_offset[list][i];

            for(j=0; j<2; j++){

                h->chroma_weight[list][16+2*i][j] = h->chroma_weight[list][16+2*i+1][j] = h->chroma_weight[list][i][j];

                h->chroma_offset[list][16+2*i][j] = h->chroma_offset[list][16+2*i+1][j] = h->chroma_offset[list][i][j];

            }

        }

    }

    for(j=0; j<h->ref_count[1]; j++){

        for(i=0; i<h->ref_count[0]; i++)

            h->implicit_weight[j][16+2*i] = h->implicit_weight[j][16+2*i+1] = h->implicit_weight[j][i];

        memcpy(h->implicit_weight[16+2*j],   h->implicit_weight[j], sizeof(*h->implicit_weight));

        memcpy(h->implicit_weight[16+2*j+1], h->implicit_weight[j], sizeof(*h->implicit_weight));

    }

}
