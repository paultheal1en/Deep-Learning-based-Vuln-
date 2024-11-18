static const uint8_t *decode_nal(H264Context *h, const uint8_t *src, int *dst_length, int *consumed, int length){

    int i, si, di;

    uint8_t *dst;

    int bufidx;



//    src[0]&0x80;                //forbidden bit

    h->nal_ref_idc= src[0]>>5;

    h->nal_unit_type= src[0]&0x1F;



    src++; length--;

#if 0

    for(i=0; i<length; i++)

        printf("%2X ", src[i]);

#endif

    for(i=0; i+1<length; i+=2){

        if(src[i]) continue;

        if(i>0 && src[i-1]==0) i--;

        if(i+2<length && src[i+1]==0 && src[i+2]<=3){

            if(src[i+2]!=3){

                /* startcode, so we must be past the end */

                length=i;

            }

            break;

        }

    }



    if(i>=length-1){ //no escaped 0

        *dst_length= length;

        *consumed= length+1; //+1 for the header

        return src;

    }



    bufidx = h->nal_unit_type == NAL_DPC ? 1 : 0; // use second escape buffer for inter data

    h->rbsp_buffer[bufidx]= av_fast_realloc(h->rbsp_buffer[bufidx], &h->rbsp_buffer_size[bufidx], length);

    dst= h->rbsp_buffer[bufidx];



    if (dst == NULL){

        return NULL;

    }



//printf("decoding esc\n");

    si=di=0;

    while(si<length){

        //remove escapes (very rare 1:2^22)

        if(si+2<length && src[si]==0 && src[si+1]==0 && src[si+2]<=3){

            if(src[si+2]==3){ //escape

                dst[di++]= 0;

                dst[di++]= 0;

                si+=3;

                continue;

            }else //next start code

                break;

        }



        dst[di++]= src[si++];

    }



    *dst_length= di;

    *consumed= si + 1;//+1 for the header

//FIXME store exact number of bits in the getbitcontext (it is needed for decoding)

    return dst;

}
