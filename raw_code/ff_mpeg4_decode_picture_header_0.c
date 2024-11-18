int ff_mpeg4_decode_picture_header(MpegEncContext * s, GetBitContext *gb)

{

    int startcode, v;



    /* search next start code */

    align_get_bits(gb);

    startcode = 0xff;

    for(;;) {

        v = get_bits(gb, 8);

        startcode = ((startcode << 8) | v) & 0xffffffff;

        

        if(get_bits_count(gb) >= gb->size*8){

            if(gb->size==1 && s->divx_version){

                printf("frame skip %d\n", gb->size);

                return FRAME_SKIPED; //divx bug

            }else

                return -1; //end of stream

        }



        if((startcode&0xFFFFFF00) != 0x100)

            continue; //no startcode

        

        if(s->avctx->debug&FF_DEBUG_STARTCODE){

            printf("startcode: %3X ", startcode);

            if     (startcode<=0x11F) printf("Video Object Start");

            else if(startcode<=0x12F) printf("Video Object Layer Start");

            else if(startcode<=0x13F) printf("Reserved");

            else if(startcode<=0x15F) printf("FGS bp start");

            else if(startcode<=0x1AF) printf("Reserved");

            else if(startcode==0x1B0) printf("Visual Object Seq Start");

            else if(startcode==0x1B1) printf("Visual Object Seq End");

            else if(startcode==0x1B2) printf("User Data");

            else if(startcode==0x1B3) printf("Group of VOP start");

            else if(startcode==0x1B4) printf("Video Session Error");

            else if(startcode==0x1B5) printf("Visual Object Start");

            else if(startcode==0x1B6) printf("Video Object Plane start");

            else if(startcode==0x1B7) printf("slice start");

            else if(startcode==0x1B8) printf("extension start");

            else if(startcode==0x1B9) printf("fgs start");

            else if(startcode==0x1BA) printf("FBA Object start");

            else if(startcode==0x1BB) printf("FBA Object Plane start");

            else if(startcode==0x1BC) printf("Mesh Object start");

            else if(startcode==0x1BD) printf("Mesh Object Plane start");

            else if(startcode==0x1BE) printf("Still Textutre Object start");

            else if(startcode==0x1BF) printf("Textutre Spatial Layer start");

            else if(startcode==0x1C0) printf("Textutre SNR Layer start");

            else if(startcode==0x1C1) printf("Textutre Tile start");

            else if(startcode==0x1C2) printf("Textutre Shape Layer start");

            else if(startcode==0x1C3) printf("stuffing start");

            else if(startcode<=0x1C5) printf("reserved");

            else if(startcode<=0x1FF) printf("System start");

            printf(" at %d\n", get_bits_count(gb));

        }



        switch(startcode){

        case 0x120:

            decode_vol_header(s, gb);

            break;

        case USER_DATA_STARTCODE:

            decode_user_data(s, gb);

            break;

        case GOP_STARTCODE:

            mpeg4_decode_gop_header(s, gb);

            break;

        case VOP_STARTCODE:

            return decode_vop_header(s, gb);

        default:

            break;

        }



        align_get_bits(gb);

        startcode = 0xff;

    }

}
