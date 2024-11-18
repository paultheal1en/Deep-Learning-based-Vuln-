static int bitplane_decoding(uint8_t* plane, int width, int height, VC9Context *v)

{

    int imode, x, y, i, code, use_vertical_tile, tile_w, tile_h;

    uint8_t invert, *planep = plane;

    int stride= width;



    invert = get_bits(&v->gb, 1);

    imode = get_vlc2(&v->gb, vc9_imode_vlc.table, VC9_IMODE_VLC_BITS, 2);

    av_log(v->avctx, AV_LOG_DEBUG, "Bitplane: imode=%i, invert=%i\n",

           imode, invert);



    switch (imode)

    {

    case IMODE_RAW:

        for (y=0; y<height; y++)

        {

            for (x=0; x<width; x++)

                planep[x] = (-get_bits(&v->gb, 1)); //-1=0xFF

            planep += stride;

        }

        invert=0; //spec says ignore invert if raw

        break;

    case IMODE_DIFF2:

    case IMODE_NORM2:

        if ((height*width) & 1) *(++planep) = get_bits(&v->gb, 1);

        for(i=0; i<(height*width)>>1; i++){

            code = get_vlc2(&v->gb, vc9_norm2_vlc.table, VC9_NORM2_VLC_BITS, 2);

            *(++planep) = code&1; //lsb => left

            *(++planep) = code&2; //msb => right - this is a bitplane, so only !0 matters

            //FIXME width->stride

        }

        break;

    case IMODE_DIFF6:

    case IMODE_NORM6:

        use_vertical_tile= height%3==0 && width%3!=0;

        tile_w= use_vertical_tile ? 2 : 3;

        tile_h= use_vertical_tile ? 3 : 2;



        for(y= height%tile_h; y<height; y+=tile_h){

            for(x= width%tile_w; x<width; x+=tile_w){

                code = get_vlc2(&v->gb, vc9_norm6_vlc.table, VC9_NORM6_VLC_BITS, 2);

                //FIXME following is a pure guess and probably wrong

                planep[x     + 0*stride]= (code>>0)&1;

                planep[x + 1 + 0*stride]= (code>>1)&1;

                if(use_vertical_tile){

                    planep[x + 0 + 1*stride]= (code>>2)&1;

                    planep[x + 1 + 1*stride]= (code>>3)&1;

                    planep[x + 0 + 2*stride]= (code>>4)&1;

                    planep[x + 1 + 2*stride]= (code>>5)&1;

                }else{

                    planep[x + 2 + 0*stride]= (code>>2)&1;

                    planep[x + 0 + 1*stride]= (code>>3)&1;

                    planep[x + 1 + 1*stride]= (code>>4)&1;

                    planep[x + 2 + 1*stride]= (code>>5)&1;

                }

            }

        }



        x= width % tile_w;

        decode_colskip(plane  ,         x, height         , stride, v);

        decode_rowskip(plane+x, width - x, height % tile_h, stride, v);



        break;

    case IMODE_ROWSKIP:

        decode_rowskip(plane, width, height, stride, v);

        break;

    case IMODE_COLSKIP: //Teh ugly

        decode_colskip(plane, width, height, stride, v);

        break;

    default: break;

    }



    /* Applying diff operator */

    if (imode == IMODE_DIFF2 || imode == IMODE_DIFF6)

    {

        planep = plane;

        planep[0] ^= invert;

        for (x=1; x<width; x++)

            planep[x] ^= planep[x-1];

        for (y=1; y<height; y++)

        {

            planep += stride;

            planep[0] ^= planep[-stride];

            for (x=1; x<width; x++)

            {

                if (planep[x-1] != planep[x-stride]) planep[x] ^= invert;

                else                                 planep[x] ^= planep[x-1];

            }

        }

    }

    else if (invert)

    {

        planep = plane;

        for (x=0; x<width*height; x++) planep[x] = !planep[x]; //FIXME stride

    }

    return 0;

}
