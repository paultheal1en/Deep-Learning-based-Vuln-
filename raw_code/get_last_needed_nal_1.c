static int get_last_needed_nal(H264Context *h)

{

    int nals_needed = 0;

    int i;



    for (i = 0; i < h->pkt.nb_nals; i++) {

        H2645NAL *nal = &h->pkt.nals[i];

        GetBitContext gb;



        /* packets can sometimes contain multiple PPS/SPS,

         * e.g. two PAFF field pictures in one packet, or a demuxer

         * which splits NALs strangely if so, when frame threading we

         * can't start the next thread until we've read all of them */

        switch (nal->type) {

        case H264_NAL_SPS:

        case H264_NAL_PPS:

            nals_needed = i;

            break;

        case H264_NAL_DPA:

        case H264_NAL_IDR_SLICE:

        case H264_NAL_SLICE:

            init_get_bits(&gb, nal->data + 1, (nal->size - 1) * 8);

            if (!get_ue_golomb(&gb))

                nals_needed = i;

        }

    }



    return nals_needed;

}
