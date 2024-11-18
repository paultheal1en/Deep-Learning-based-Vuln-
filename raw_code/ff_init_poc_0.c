int ff_init_poc(H264Context *h, int pic_field_poc[2], int *pic_poc)

{

    const SPS *sps = h->ps.sps;

    const int max_frame_num = 1 << sps->log2_max_frame_num;

    int field_poc[2];



    h->frame_num_offset = h->prev_frame_num_offset;

    if (h->frame_num < h->prev_frame_num)

        h->frame_num_offset += max_frame_num;



    if (sps->poc_type == 0) {

        const int max_poc_lsb = 1 << sps->log2_max_poc_lsb;



        if (h->poc_lsb < h->prev_poc_lsb &&

            h->prev_poc_lsb - h->poc_lsb >= max_poc_lsb / 2)

            h->poc_msb = h->prev_poc_msb + max_poc_lsb;

        else if (h->poc_lsb > h->prev_poc_lsb &&

                 h->prev_poc_lsb - h->poc_lsb < -max_poc_lsb / 2)

            h->poc_msb = h->prev_poc_msb - max_poc_lsb;

        else

            h->poc_msb = h->prev_poc_msb;

        field_poc[0] =

        field_poc[1] = h->poc_msb + h->poc_lsb;

        if (h->picture_structure == PICT_FRAME)

            field_poc[1] += h->delta_poc_bottom;

    } else if (sps->poc_type == 1) {

        int abs_frame_num, expected_delta_per_poc_cycle, expectedpoc;

        int i;



        if (sps->poc_cycle_length != 0)

            abs_frame_num = h->frame_num_offset + h->frame_num;

        else

            abs_frame_num = 0;



        if (h->nal_ref_idc == 0 && abs_frame_num > 0)

            abs_frame_num--;



        expected_delta_per_poc_cycle = 0;

        for (i = 0; i < sps->poc_cycle_length; i++)

            // FIXME integrate during sps parse

            expected_delta_per_poc_cycle += sps->offset_for_ref_frame[i];



        if (abs_frame_num > 0) {

            int poc_cycle_cnt          = (abs_frame_num - 1) / sps->poc_cycle_length;

            int frame_num_in_poc_cycle = (abs_frame_num - 1) % sps->poc_cycle_length;



            expectedpoc = poc_cycle_cnt * expected_delta_per_poc_cycle;

            for (i = 0; i <= frame_num_in_poc_cycle; i++)

                expectedpoc = expectedpoc + sps->offset_for_ref_frame[i];

        } else

            expectedpoc = 0;



        if (h->nal_ref_idc == 0)

            expectedpoc = expectedpoc + sps->offset_for_non_ref_pic;



        field_poc[0] = expectedpoc + h->delta_poc[0];

        field_poc[1] = field_poc[0] + sps->offset_for_top_to_bottom_field;



        if (h->picture_structure == PICT_FRAME)

            field_poc[1] += h->delta_poc[1];

    } else {

        int poc = 2 * (h->frame_num_offset + h->frame_num);



        if (!h->nal_ref_idc)

            poc--;



        field_poc[0] = poc;

        field_poc[1] = poc;

    }



    if (h->picture_structure != PICT_BOTTOM_FIELD)

        pic_field_poc[0] = field_poc[0];

    if (h->picture_structure != PICT_TOP_FIELD)

        pic_field_poc[1] = field_poc[1];

    *pic_poc = FFMIN(pic_field_poc[0], pic_field_poc[1]);



    return 0;

}
