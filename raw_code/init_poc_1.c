static int init_poc(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int max_frame_num= 1<<h->sps.log2_max_frame_num;

    int field_poc[2];



    h->frame_num_offset= h->prev_frame_num_offset;

    if(h->frame_num < h->prev_frame_num)

        h->frame_num_offset += max_frame_num;



    if(h->sps.poc_type==0){

        const int max_poc_lsb= 1<<h->sps.log2_max_poc_lsb;



        if     (h->poc_lsb < h->prev_poc_lsb && h->prev_poc_lsb - h->poc_lsb >= max_poc_lsb/2)

            h->poc_msb = h->prev_poc_msb + max_poc_lsb;

        else if(h->poc_lsb > h->prev_poc_lsb && h->prev_poc_lsb - h->poc_lsb < -max_poc_lsb/2)

            h->poc_msb = h->prev_poc_msb - max_poc_lsb;

        else

            h->poc_msb = h->prev_poc_msb;

//printf("poc: %d %d\n", h->poc_msb, h->poc_lsb);

        field_poc[0] =

        field_poc[1] = h->poc_msb + h->poc_lsb;

        if(s->picture_structure == PICT_FRAME)

            field_poc[1] += h->delta_poc_bottom;

    }else if(h->sps.poc_type==1){

        int abs_frame_num, expected_delta_per_poc_cycle, expectedpoc;

        int i;



        if(h->sps.poc_cycle_length != 0)

            abs_frame_num = h->frame_num_offset + h->frame_num;

        else

            abs_frame_num = 0;



        if(h->nal_ref_idc==0 && abs_frame_num > 0)

            abs_frame_num--;



        expected_delta_per_poc_cycle = 0;

        for(i=0; i < h->sps.poc_cycle_length; i++)

            expected_delta_per_poc_cycle += h->sps.offset_for_ref_frame[ i ]; //FIXME integrate during sps parse



        if(abs_frame_num > 0){

            int poc_cycle_cnt          = (abs_frame_num - 1) / h->sps.poc_cycle_length;

            int frame_num_in_poc_cycle = (abs_frame_num - 1) % h->sps.poc_cycle_length;



            expectedpoc = poc_cycle_cnt * expected_delta_per_poc_cycle;

            for(i = 0; i <= frame_num_in_poc_cycle; i++)

                expectedpoc = expectedpoc + h->sps.offset_for_ref_frame[ i ];

        } else

            expectedpoc = 0;



        if(h->nal_ref_idc == 0)

            expectedpoc = expectedpoc + h->sps.offset_for_non_ref_pic;



        field_poc[0] = expectedpoc + h->delta_poc[0];

        field_poc[1] = field_poc[0] + h->sps.offset_for_top_to_bottom_field;



        if(s->picture_structure == PICT_FRAME)

            field_poc[1] += h->delta_poc[1];

    }else{

        int poc= 2*(h->frame_num_offset + h->frame_num);



        if(!h->nal_ref_idc)

            poc--;



        field_poc[0]= poc;

        field_poc[1]= poc;

    }



    if(s->picture_structure != PICT_BOTTOM_FIELD) {

        s->current_picture_ptr->field_poc[0]= field_poc[0];

        s->current_picture_ptr->poc = field_poc[0];

    }

    if(s->picture_structure != PICT_TOP_FIELD) {

        s->current_picture_ptr->field_poc[1]= field_poc[1];

        s->current_picture_ptr->poc = field_poc[1];

    }

    if(!FIELD_PICTURE || !s->first_field) {

        Picture *cur = s->current_picture_ptr;

        cur->poc= FFMIN(cur->field_poc[0], cur->field_poc[1]);

    }



    return 0;

}
