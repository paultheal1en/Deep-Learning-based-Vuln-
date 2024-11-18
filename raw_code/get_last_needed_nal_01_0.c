static int get_last_needed_nal(H264Context *h, const uint8_t *buf, int buf_size)

{

    int next_avc    = h->is_avc ? 0 : buf_size;

    int nal_index   = 0;

    int buf_index   = 0;

    int nals_needed = 0;



    while(1) {

        int nalsize = 0;

        int dst_length, bit_length, consumed;

        const uint8_t *ptr;



        if (buf_index >= next_avc) {

            nalsize = get_avc_nalsize(h, buf, buf_size, &buf_index);

            if (nalsize < 0)

                break;

            next_avc = buf_index + nalsize;

        } else {

            buf_index = find_start_code(buf, buf_size, buf_index, next_avc);

            if (buf_index >= buf_size)

                break;

        }



        ptr = ff_h264_decode_nal(h, buf + buf_index, &dst_length, &consumed,

                                 next_avc - buf_index);



        if (ptr == NULL || dst_length < 0)

            return AVERROR_INVALIDDATA;



        buf_index += consumed;



        bit_length = get_bit_length(h, buf, ptr, dst_length,

                                    buf_index, next_avc);

        nal_index++;



        /* packets can sometimes contain multiple PPS/SPS,

         * e.g. two PAFF field pictures in one packet, or a demuxer

         * which splits NALs strangely if so, when frame threading we

         * can't start the next thread until we've read all of them */

        switch (h->nal_unit_type) {

        case NAL_SPS:

        case NAL_PPS:

            nals_needed = nal_index;

            break;

        case NAL_DPA:

        case NAL_IDR_SLICE:

        case NAL_SLICE:

            init_get_bits(&h->gb, ptr, bit_length);

            if (!get_ue_golomb(&h->gb))

                nals_needed = nal_index;

        }

    }



    return nals_needed;

}
