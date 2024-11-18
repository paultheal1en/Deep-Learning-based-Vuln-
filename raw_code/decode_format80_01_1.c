static void decode_format80(const unsigned char *src, int src_size,

    unsigned char *dest, int dest_size, int check_size) {



    int src_index = 0;

    int dest_index = 0;

    int count;

    int src_pos;

    unsigned char color;

    int i;



    while (src_index < src_size) {



        av_dlog(NULL, "      opcode %02X: ", src[src_index]);



        /* 0x80 means that frame is finished */

        if (src[src_index] == 0x80)




        if (dest_index >= dest_size) {

            av_log(NULL, AV_LOG_ERROR, "  VQA video: decode_format80 problem: dest_index (%d) exceeded dest_size (%d)\n",

                dest_index, dest_size);


        }



        if (src[src_index] == 0xFF) {



            src_index++;

            count = AV_RL16(&src[src_index]);

            src_index += 2;

            src_pos = AV_RL16(&src[src_index]);

            src_index += 2;

            av_dlog(NULL, "(1) copy %X bytes from absolute pos %X\n", count, src_pos);

            CHECK_COUNT();

            if (src_pos + count > dest_size)


            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[src_pos + i];

            dest_index += count;



        } else if (src[src_index] == 0xFE) {



            src_index++;

            count = AV_RL16(&src[src_index]);

            src_index += 2;

            color = src[src_index++];

            av_dlog(NULL, "(2) set %X bytes to %02X\n", count, color);

            CHECK_COUNT();

            memset(&dest[dest_index], color, count);

            dest_index += count;



        } else if ((src[src_index] & 0xC0) == 0xC0) {



            count = (src[src_index++] & 0x3F) + 3;

            src_pos = AV_RL16(&src[src_index]);

            src_index += 2;

            av_dlog(NULL, "(3) copy %X bytes from absolute pos %X\n", count, src_pos);

            CHECK_COUNT();

            if (src_pos + count > dest_size)


            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[src_pos + i];

            dest_index += count;



        } else if (src[src_index] > 0x80) {



            count = src[src_index++] & 0x3F;

            av_dlog(NULL, "(4) copy %X bytes from source to dest\n", count);

            CHECK_COUNT();

            memcpy(&dest[dest_index], &src[src_index], count);

            src_index += count;

            dest_index += count;



        } else {



            count = ((src[src_index] & 0x70) >> 4) + 3;

            src_pos = AV_RB16(&src[src_index]) & 0x0FFF;

            src_index += 2;

            av_dlog(NULL, "(5) copy %X bytes from relpos %X\n", count, src_pos);

            CHECK_COUNT();



            for (i = 0; i < count; i++)

                dest[dest_index + i] = dest[dest_index - src_pos + i];

            dest_index += count;

        }

    }



    /* validate that the entire destination buffer was filled; this is

     * important for decoding frame maps since each vector needs to have a

     * codebook entry; it is not important for compressed codebooks because

     * not every entry needs to be filled */

    if (check_size)

        if (dest_index < dest_size)

            av_log(NULL, AV_LOG_ERROR, "  VQA video: decode_format80 problem: decode finished with dest_index (%d) < dest_size (%d)\n",

                dest_index, dest_size);

}