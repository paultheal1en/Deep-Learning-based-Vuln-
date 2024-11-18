static int x8_setup_spatial_predictor(IntraX8Context * const w, const int chroma){

    MpegEncContext * const s= w->s;

    int range;

    int sum;

    int quant;



    w->dsp.setup_spatial_compensation(s->dest[chroma], s->edge_emu_buffer,

                                      s->current_picture.f.linesize[chroma>0],

                                      &range, &sum, w->edges);

    if(chroma){

        w->orient=w->chroma_orient;

        quant=w->quant_dc_chroma;

    }else{

        quant=w->quant;

    }



    w->flat_dc=0;

    if(range < quant || range < 3){

        w->orient=0;

        if(range < 3){//yep you read right, a +-1 idct error may break decoding!

            w->flat_dc=1;

            sum+=9;

            w->predicted_dc = (sum*6899)>>17;//((1<<17)+9)/(8+8+1+2)=6899

        }

    }

    if(chroma)

        return 0;



    assert(w->orient < 3);

    if(range < 2*w->quant){

        if( (w->edges&3) == 0){

            if(w->orient==1) w->orient=11;

            if(w->orient==2) w->orient=10;

        }else{

            w->orient=0;

        }

        w->raw_orient=0;

    }else{

        static const uint8_t prediction_table[3][12]={

            {0,8,4, 10,11, 2,6,9,1,3,5,7},

            {4,0,8, 11,10, 3,5,2,6,9,1,7},

            {8,0,4, 10,11, 1,7,2,6,9,3,5}

        };

        w->raw_orient=x8_get_orient_vlc(w);

        if(w->raw_orient<0) return -1;

        assert(w->raw_orient < 12 );

        assert(w->orient<3);

        w->orient=prediction_table[w->orient][w->raw_orient];

    }

    return 0;

}
