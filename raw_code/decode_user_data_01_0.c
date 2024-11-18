static int decode_user_data(MpegEncContext *s, GetBitContext *gb){

    char buf[256];

    int i;

    int e;

    int ver = 0, build = 0, ver2 = 0, ver3 = 0;

    char last;



    for(i=0; i<255 && get_bits_count(gb) < gb->size_in_bits; i++){

        if(show_bits(gb, 23) == 0) break;

        buf[i]= get_bits(gb, 8);

    }

    buf[i]=0;



    /* divx detection */

    e=sscanf(buf, "DivX%dBuild%d%c", &ver, &build, &last);

    if(e<2)

        e=sscanf(buf, "DivX%db%d%c", &ver, &build, &last);

    if(e>=2){

        s->divx_version= ver;

        s->divx_build= build;

        s->divx_packed= e==3 && last=='p';

        if(s->divx_packed)

            av_log(s->avctx, AV_LOG_WARNING, "Invalid and inefficient vfw-avi packed B frames detected\n");

    }



    /* ffmpeg detection */

    e=sscanf(buf, "FFmpe%*[^b]b%d", &build)+3;

    if(e!=4)

        e=sscanf(buf, "FFmpeg v%d.%d.%d / libavcodec build: %d", &ver, &ver2, &ver3, &build);

    if(e!=4){

        e=sscanf(buf, "Lavc%d.%d.%d", &ver, &ver2, &ver3)+1;

        if (e>1)

            build= (ver<<16) + (ver2<<8) + ver3;

    }

    if(e!=4){

        if(strcmp(buf, "ffmpeg")==0){

            s->lavc_build= 4600;

        }

    }

    if(e==4){

        s->lavc_build= build;

    }



    /* Xvid detection */

    e=sscanf(buf, "XviD%d", &build);

    if(e==1){

        s->xvid_build= build;

    }



//printf("User Data: %s\n", buf);

    return 0;

}
