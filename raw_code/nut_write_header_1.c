static int nut_write_header(AVFormatContext *s)

{

    NUTContext *nut = s->priv_data;

    ByteIOContext *bc = &s->pb;

    AVCodecContext *codec;

    int i, j, tmp_time, tmp_flags,tmp_stream, tmp_mul, tmp_size, tmp_fields;



    nut->avf= s;

    

    nut->stream =	

	av_mallocz(sizeof(StreamContext)*s->nb_streams);

        



    put_buffer(bc, ID_STRING, strlen(ID_STRING));

    put_byte(bc, 0);

    nut->packet_start[2]= url_ftell(bc);

    

    /* main header */

    put_be64(bc, MAIN_STARTCODE);

    put_packetheader(nut, bc, 120+5*256, 1);

    put_v(bc, 2); /* version */

    put_v(bc, s->nb_streams);

    put_v(bc, MAX_DISTANCE);

    put_v(bc, MAX_SHORT_DISTANCE);

    

    put_v(bc, nut->rate_num=1);

    put_v(bc, nut->rate_den=2);

    put_v(bc, nut->short_startcode=0x4EFE79);

    

    build_frame_code(s);

    assert(nut->frame_code['N'].flags == FLAG_INVALID);

    

    tmp_time= tmp_flags= tmp_stream= tmp_mul= tmp_size= /*tmp_res=*/ INT_MAX;

    for(i=0; i<256;){

        tmp_fields=0;

        tmp_size= 0;

        if(tmp_time   != nut->frame_code[i].timestamp_delta) tmp_fields=1;

        if(tmp_mul    != nut->frame_code[i].size_mul       ) tmp_fields=2;

        if(tmp_stream != nut->frame_code[i].stream_id_plus1) tmp_fields=3;

        if(tmp_size   != nut->frame_code[i].size_lsb       ) tmp_fields=4;

//        if(tmp_res    != nut->frame_code[i].res            ) tmp_fields=5;



        tmp_time  = nut->frame_code[i].timestamp_delta;

        tmp_flags = nut->frame_code[i].flags;

        tmp_stream= nut->frame_code[i].stream_id_plus1;

        tmp_mul   = nut->frame_code[i].size_mul;

        tmp_size  = nut->frame_code[i].size_lsb;

//        tmp_res   = nut->frame_code[i].res;

        

        for(j=0; i<256; j++,i++){

            if(nut->frame_code[i].timestamp_delta != tmp_time  ) break;

            if(nut->frame_code[i].flags           != tmp_flags ) break;

            if(nut->frame_code[i].stream_id_plus1 != tmp_stream) break;

            if(nut->frame_code[i].size_mul        != tmp_mul   ) break;

            if(nut->frame_code[i].size_lsb        != tmp_size+j) break;

//            if(nut->frame_code[i].res             != tmp_res   ) break;

        }

        if(j != tmp_mul - tmp_size) tmp_fields=6;



        put_v(bc, tmp_flags);

        put_v(bc, tmp_fields);

        if(tmp_fields>0) put_s(bc, tmp_time);

        if(tmp_fields>1) put_v(bc, tmp_mul);

        if(tmp_fields>2) put_v(bc, tmp_stream);

        if(tmp_fields>3) put_v(bc, tmp_size);

        if(tmp_fields>4) put_v(bc, 0 /*tmp_res*/);

        if(tmp_fields>5) put_v(bc, j);

    }



    update_packetheader(nut, bc, 0, 1);

    

    /* stream headers */

    for (i = 0; i < s->nb_streams; i++)

    {

	int nom, denom, gcd;



	codec = &s->streams[i]->codec;

	

	put_be64(bc, STREAM_STARTCODE);

	put_packetheader(nut, bc, 120 + codec->extradata_size, 1);

	put_v(bc, i /*s->streams[i]->index*/);

	put_v(bc, (codec->codec_type == CODEC_TYPE_AUDIO) ? 32 : 0);

	if (codec->codec_tag)

	    put_vb(bc, codec->codec_tag);

	else if (codec->codec_type == CODEC_TYPE_VIDEO)

	{

	    put_vb(bc, codec_get_bmp_tag(codec->codec_id));

	}

	else if (codec->codec_type == CODEC_TYPE_AUDIO)

	{

	    put_vb(bc, codec_get_wav_tag(codec->codec_id));

	}

        else

            put_vb(bc, 0);



	if (codec->codec_type == CODEC_TYPE_VIDEO)

	{

	    nom = codec->time_base.den;

	    denom = codec->time_base.num;

	}

	else

	{

	    nom = codec->sample_rate;

            if(codec->frame_size>0)

                denom= codec->frame_size;

            else

                denom= 1; //unlucky

	}

        gcd= ff_gcd(nom, denom);

        nom   /= gcd;

        denom /= gcd;

        nut->stream[i].rate_num= nom;

        nut->stream[i].rate_den= denom;

        av_set_pts_info(s->streams[i], 60, denom, nom);



	put_v(bc, codec->bit_rate);

	put_vb(bc, 0); /* no language code */

	put_v(bc, nom);

	put_v(bc, denom);

        if(nom / denom < 1000)

	    nut->stream[i].msb_timestamp_shift = 7;

        else

	    nut->stream[i].msb_timestamp_shift = 14;

	put_v(bc, nut->stream[i].msb_timestamp_shift);

        put_v(bc, codec->has_b_frames);

	put_byte(bc, 0); /* flags: 0x1 - fixed_fps, 0x2 - index_present */

	

        if(codec->extradata_size){

            put_v(bc, 1);

            put_v(bc, codec->extradata_size);

            put_buffer(bc, codec->extradata, codec->extradata_size);            

        }

	put_v(bc, 0); /* end of codec specific headers */

	

	switch(codec->codec_type)

	{

	    case CODEC_TYPE_AUDIO:

		put_v(bc, codec->sample_rate);

		put_v(bc, 1);

		put_v(bc, codec->channels);

		break;

	    case CODEC_TYPE_VIDEO:

		put_v(bc, codec->width);

		put_v(bc, codec->height);

		put_v(bc, codec->sample_aspect_ratio.num);

		put_v(bc, codec->sample_aspect_ratio.den);

		put_v(bc, 0); /* csp type -- unknown */

		break;

            default:

                break;

	}

        update_packetheader(nut, bc, 0, 1);

    }



    /* info header */

    put_be64(bc, INFO_STARTCODE);

    put_packetheader(nut, bc, 30+strlen(s->author)+strlen(s->title)+

        strlen(s->comment)+strlen(s->copyright)+strlen(LIBAVFORMAT_IDENT), 1); 

    if (s->author[0])

    {

        put_v(bc, 9); /* type */

        put_str(bc, s->author);

    }

    if (s->title[0])

    {

        put_v(bc, 10); /* type */

        put_str(bc, s->title);

    }

    if (s->comment[0])

    {

        put_v(bc, 11); /* type */

        put_str(bc, s->comment);

    }

    if (s->copyright[0])

    {

        put_v(bc, 12); /* type */

        put_str(bc, s->copyright);

    }

    /* encoder */

    if(!(s->streams[0]->codec.flags & CODEC_FLAG_BITEXACT)){

        put_v(bc, 13); /* type */

        put_str(bc, LIBAVFORMAT_IDENT);

    }

    

    put_v(bc, 0); /* eof info */

    update_packetheader(nut, bc, 0, 1);

        

    put_flush_packet(bc);

    

    return 0;

}
