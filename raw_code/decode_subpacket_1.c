static int decode_subpacket(COOKContext *q, uint8_t *inbuffer,

                            int sub_packet_size, int16_t *outbuffer) {

    int i,j;

    int value;

    float* tmp_ptr;



    /* packet dump */

//    for (i=0 ; i<sub_packet_size ; i++) {

//        av_log(NULL, AV_LOG_ERROR, "%02x", inbuffer[i]);

//    }

//    av_log(NULL, AV_LOG_ERROR, "\n");



    decode_bytes(inbuffer, q->decoded_bytes_buffer, sub_packet_size);

    init_get_bits(&q->gb, q->decoded_bytes_buffer, sub_packet_size*8);

    decode_gain_info(&q->gb, &q->gain_current);

    memcpy(&q->gain_copy, &q->gain_current ,sizeof(COOKgain));  //This copy does not seem to be used. FIXME



    if(q->nb_channels==2 && q->joint_stereo==1){

        joint_decode(q, q->decode_buf_ptr[0], q->decode_buf_ptr[2]);



        /* Swap buffer pointers. */

        tmp_ptr = q->decode_buf_ptr[1];

        q->decode_buf_ptr[1] = q->decode_buf_ptr[0];

        q->decode_buf_ptr[0] = tmp_ptr;

        tmp_ptr = q->decode_buf_ptr[3];

        q->decode_buf_ptr[3] = q->decode_buf_ptr[2];

        q->decode_buf_ptr[2] = tmp_ptr;



        /* FIXME: Rethink the gainbuffer handling, maybe a rename?

           now/previous swap */

        q->gain_now_ptr = &q->gain_now;

        q->gain_previous_ptr = &q->gain_previous;

        for (i=0 ; i<q->nb_channels ; i++){



            cook_imlt(q, q->decode_buf_ptr[i*2], q->mono_mdct_output, q->mlt_tmp);

            gain_compensate(q, q->mono_mdct_output, q->gain_now_ptr,

                            q->gain_previous_ptr, q->previous_buffer_ptr[0]);



            /* Swap out the previous buffer. */

            tmp_ptr = q->previous_buffer_ptr[0];

            q->previous_buffer_ptr[0] = q->previous_buffer_ptr[1];

            q->previous_buffer_ptr[1] = tmp_ptr;



            /* Clip and convert the floats to 16 bits. */

            for (j=0 ; j<q->samples_per_frame ; j++){

                value = lrintf(q->mono_mdct_output[j]);

                if(value < -32768) value = -32768;

                else if(value > 32767) value = 32767;

                outbuffer[2*j+i] = value;

            }

        }



        memcpy(&q->gain_now, &q->gain_previous, sizeof(COOKgain));

        memcpy(&q->gain_previous, &q->gain_current, sizeof(COOKgain));



    } else if (q->nb_channels==2 && q->joint_stereo==0) {

            /* channel 0 */

            mono_decode(q, q->decode_buf_ptr[0]);



            tmp_ptr = q->decode_buf_ptr[0];

            q->decode_buf_ptr[0] = q->decode_buf_ptr[1];

            q->decode_buf_ptr[1] = q->decode_buf_ptr[2];

            q->decode_buf_ptr[2] = q->decode_buf_ptr[3];

            q->decode_buf_ptr[3] = tmp_ptr;



            q->gain_now_ptr = &q->gain_now;

            q->gain_previous_ptr = &q->gain_previous;



            cook_imlt(q, q->decode_buf_ptr[0], q->mono_mdct_output,q->mlt_tmp);

            gain_compensate(q, q->mono_mdct_output, q->gain_now_ptr,

                            q->gain_previous_ptr, q->previous_buffer_ptr[0]);

            /* Swap out the previous buffer. */

            tmp_ptr = q->previous_buffer_ptr[0];

            q->previous_buffer_ptr[0] = q->previous_buffer_ptr[1];

            q->previous_buffer_ptr[1] = tmp_ptr;



            for (j=0 ; j<q->samples_per_frame ; j++){

                value = lrintf(q->mono_mdct_output[j]);

                if(value < -32768) value = -32768;

                else if(value > 32767) value = 32767;

                outbuffer[2*j+1] = value;

            }



            /* channel 1 */

            //av_log(NULL,AV_LOG_ERROR,"bits = %d\n",get_bits_count(&q->gb));

            init_get_bits(&q->gb, q->decoded_bytes_buffer, sub_packet_size*8+q->bits_per_subpacket);

            decode_gain_info(&q->gb, &q->gain_current);

            //memcpy(&q->gain_copy, &q->gain_current ,sizeof(COOKgain));

            mono_decode(q, q->decode_buf_ptr[0]);

            tmp_ptr = q->decode_buf_ptr[0];

            q->decode_buf_ptr[1] = q->decode_buf_ptr[2];

            q->decode_buf_ptr[2] = q->decode_buf_ptr[3];

            q->decode_buf_ptr[3] = tmp_ptr;



            q->gain_now_ptr = &q->gain_now;

            q->gain_previous_ptr = &q->gain_previous;



            cook_imlt(q, q->decode_buf_ptr[0], q->mono_mdct_output,q->mlt_tmp);

            gain_compensate(q, q->mono_mdct_output, q->gain_now_ptr, q->gain_previous_ptr, q->previous_buffer_ptr[0]);



            /* Swap out the previous buffer. */

            tmp_ptr = q->previous_buffer_ptr[0];

            q->previous_buffer_ptr[0] = q->previous_buffer_ptr[1];

            q->previous_buffer_ptr[1] = tmp_ptr;



            for (j=0 ; j<q->samples_per_frame ; j++){

                value = lrintf(q->mono_mdct_output[j]);

                if(value < -32768) value = -32768;

                else if(value > 32767) value = 32767;

                outbuffer[2*j] = value;

            }





            /* Swap out the previous buffer. */

            memcpy(&q->gain_now, &q->gain_previous, sizeof(COOKgain));

            memcpy(&q->gain_previous, &q->gain_current, sizeof(COOKgain));



    } else {

        mono_decode(q, q->decode_buf_ptr[0]);



        /* Swap buffer pointers. */

        tmp_ptr = q->decode_buf_ptr[1];

        q->decode_buf_ptr[1] = q->decode_buf_ptr[0];

        q->decode_buf_ptr[0] = tmp_ptr;



        /* FIXME: Rethink the gainbuffer handling, maybe a rename?

           now/previous swap */

        q->gain_now_ptr = &q->gain_now;

        q->gain_previous_ptr = &q->gain_previous;



        cook_imlt(q, q->decode_buf_ptr[0], q->mono_mdct_output,q->mlt_tmp);

        gain_compensate(q, q->mono_mdct_output, q->gain_now_ptr,

                        q->gain_previous_ptr, q->mono_previous_buffer1);



        /* Clip and convert the floats to 16 bits */

        for (j=0 ; j<q->samples_per_frame ; j++){

            value = lrintf(q->mono_mdct_output[j]);

            if(value < -32768) value = -32768;

            else if(value > 32767) value = 32767;

            outbuffer[j] = value;

        }

        memcpy(&q->gain_now, &q->gain_previous, sizeof(COOKgain));

        memcpy(&q->gain_previous, &q->gain_current, sizeof(COOKgain));

    }

    return q->samples_per_frame * sizeof(int16_t);

}
