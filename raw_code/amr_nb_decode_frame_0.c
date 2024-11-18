static int amr_nb_decode_frame(AVCodecContext * avctx,

            void *data, int *data_size,

            uint8_t * buf, int buf_size)

{

    AMRContext *s = avctx->priv_data;

    uint8_t*amrData=buf;

    int offset=0;

    UWord8 toc, q, ft;

    Word16 serial[SERIAL_FRAMESIZE];   /* coded bits */

    Word16 *synth;

    UWord8 *packed_bits;

    static Word16 packed_size[16] = {12, 13, 15, 17, 19, 20, 26, 31, 5, 0, 0, 0, 0, 0, 0, 0};

    int i;



    //printf("amr_decode_frame data_size=%i buf=0x%X buf_size=%d frameCount=%d!!\n",*data_size,buf,buf_size,s->frameCount);



    synth=data;



//    while(offset<buf_size)

    {

        toc=amrData[offset];

        /* read rest of the frame based on ToC byte */

        q  = (toc >> 2) & 0x01;

        ft = (toc >> 3) & 0x0F;



        //printf("offset=%d, packet_size=%d amrData= 0x%X %X %X %X\n",offset,packed_size[ft],amrData[offset],amrData[offset+1],amrData[offset+2],amrData[offset+3]);



        offset++;



        packed_bits=amrData+offset;



        offset+=packed_size[ft];



        //Unsort and unpack bits

        s->rx_type = UnpackBits(q, ft, packed_bits, &s->mode, &serial[1]);



        //We have a new frame

        s->frameCount++;



        if (s->rx_type == RX_NO_DATA)

        {

            s->mode = s->speech_decoder_state->prev_mode;

        }

        else {

            s->speech_decoder_state->prev_mode = s->mode;

        }



        /* if homed: check if this frame is another homing frame */

        if (s->reset_flag_old == 1)

        {

            /* only check until end of first subframe */

            s->reset_flag = decoder_homing_frame_test_first(&serial[1], s->mode);

        }

        /* produce encoder homing frame if homed & input=decoder homing frame */

        if ((s->reset_flag != 0) && (s->reset_flag_old != 0))

        {

            for (i = 0; i < L_FRAME; i++)

            {

                synth[i] = EHF_MASK;

            }

        }

        else

        {

            /* decode frame */

            Speech_Decode_Frame(s->speech_decoder_state, s->mode, &serial[1], s->rx_type, synth);

        }



        //Each AMR-frame results in 160 16-bit samples

        *data_size+=160*2;

        synth+=160;



        /* if not homed: check whether current frame is a homing frame */

        if (s->reset_flag_old == 0)

        {

            /* check whole frame */

            s->reset_flag = decoder_homing_frame_test(&serial[1], s->mode);

        }

        /* reset decoder if current frame is a homing frame */

        if (s->reset_flag != 0)

        {

            Speech_Decode_Frame_reset(s->speech_decoder_state);

        }

        s->reset_flag_old = s->reset_flag;



    }

    return offset;

}
