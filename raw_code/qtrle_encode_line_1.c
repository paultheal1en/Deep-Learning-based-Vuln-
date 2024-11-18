static void qtrle_encode_line(QtrleEncContext *s, AVFrame *p, int line, uint8_t **buf)

{

    int width=s->logical_width;

    int i;

    signed char rlecode;



    /* We will use it to compute the best bulk copy sequence */

    unsigned int bulkcount;

    /* This will be the number of pixels equal to the preivous frame one's

     * starting from the ith pixel */

    unsigned int skipcount;

    /* This will be the number of consecutive equal pixels in the current

     * frame, starting from the ith one also */

    unsigned int repeatcount;



    /* The cost of the three different possibilities */

    int total_bulk_cost;

    int total_skip_cost;

    int total_repeat_cost;



    int temp_cost;

    int j;



    uint8_t *this_line = p->               data[0] + line*p->               linesize[0] +

        (width - 1)*s->pixel_size;

    uint8_t *prev_line = s->previous_frame.data[0] + line*s->previous_frame.linesize[0] +

        (width - 1)*s->pixel_size;



    s->length_table[width] = 0;

    skipcount = 0;



    for (i = width - 1; i >= 0; i--) {



        if (!s->frame.key_frame && !memcmp(this_line, prev_line, s->pixel_size))

            skipcount = FFMIN(skipcount + 1, MAX_RLE_SKIP);

        else

            skipcount = 0;



        total_skip_cost  = s->length_table[i + skipcount] + 2;

        s->skip_table[i] = skipcount;





        if (i < width - 1 && !memcmp(this_line, this_line + s->pixel_size, s->pixel_size))

            repeatcount = FFMIN(repeatcount + 1, MAX_RLE_REPEAT);

        else

            repeatcount = 1;



        total_repeat_cost = s->length_table[i + repeatcount] + 1 + s->pixel_size;



        /* skip code is free for the first pixel, it costs one byte for repeat and bulk copy

         * so let's make it aware */

        if (i == 0) {

            total_skip_cost--;

            total_repeat_cost++;

        }



        if (repeatcount > 1 && (skipcount == 0 || total_repeat_cost < total_skip_cost)) {

            /* repeat is the best */

            s->length_table[i]  = total_repeat_cost;

            s->rlecode_table[i] = -repeatcount;

        }

        else if (skipcount > 0) {

            /* skip is the best choice here */

            s->length_table[i]  = total_skip_cost;

            s->rlecode_table[i] = 0;

        }

        else {

            /* We cannot do neither skip nor repeat

             * thus we search for the best bulk copy to do */



            int limit = FFMIN(width - i, MAX_RLE_BULK);



            temp_cost = 1 + s->pixel_size + !i;

            total_bulk_cost = INT_MAX;



            for (j = 1; j <= limit; j++) {

                if (s->length_table[i + j] + temp_cost < total_bulk_cost) {

                    /* We have found a better bulk copy ... */

                    total_bulk_cost = s->length_table[i + j] + temp_cost;

                    bulkcount = j;

                }

                temp_cost += s->pixel_size;

            }



            s->length_table[i]  = total_bulk_cost;

            s->rlecode_table[i] = bulkcount;

        }



        this_line -= s->pixel_size;

        prev_line -= s->pixel_size;

    }



    /* Good ! Now we have the best sequence for this line, let's output it */



    /* We do a special case for the first pixel so that we avoid testing it in

     * the whole loop */



    i=0;

    this_line = p->               data[0] + line*p->linesize[0];



    if (s->rlecode_table[0] == 0) {

        bytestream_put_byte(buf, s->skip_table[0] + 1);

        i += s->skip_table[0];

    }

    else bytestream_put_byte(buf, 1);





    while (i < width) {

        rlecode = s->rlecode_table[i];

        bytestream_put_byte(buf, rlecode);

        if (rlecode == 0) {

            /* Write a skip sequence */

            bytestream_put_byte(buf, s->skip_table[i] + 1);

            i += s->skip_table[i];

        }

        else if (rlecode > 0) {

            /* bulk copy */

            if (s->avctx->pix_fmt == PIX_FMT_GRAY8) {

                int j;

                // QT grayscale colorspace has 0=white and 255=black, we will

                // ignore the palette that is included in the AVFrame because

                // PIX_FMT_GRAY8 has defined color mapping

                for (j = 0; j < rlecode*s->pixel_size; ++j)

                    bytestream_put_byte(buf, *(this_line + i*s->pixel_size + j) ^ 0xff);

            } else {

                bytestream_put_buffer(buf, this_line + i*s->pixel_size, rlecode*s->pixel_size);

            }

            i += rlecode;

        }

        else {

            /* repeat the bits */

            if (s->avctx->pix_fmt == PIX_FMT_GRAY8) {

                int j;

                // QT grayscale colorspace has 0=white and 255=black, ...

                for (j = 0; j < s->pixel_size; ++j)

                    bytestream_put_byte(buf, *(this_line + i*s->pixel_size + j) ^ 0xff);

            } else {

                bytestream_put_buffer(buf, this_line + i*s->pixel_size, s->pixel_size);

            }

            i -= rlecode;

        }

    }

    bytestream_put_byte(buf, -1); // end RLE line

}
