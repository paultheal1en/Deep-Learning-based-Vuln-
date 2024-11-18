static int mov_read_stsd(MOVContext *c, ByteIOContext *pb, MOV_atom_t atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    MOVStreamContext *sc = (MOVStreamContext *)st->priv_data;

    int entries, frames_per_sample;

    uint32_t format;



    print_atom("stsd", atom);



    get_byte(pb); /* version */

    get_byte(pb); get_byte(pb); get_byte(pb); /* flags */



    entries = get_be32(pb);



    while(entries--) {

        enum CodecID id;

	int size = get_be32(pb); /* size */

        format = get_le32(pb); /* data format */



        get_be32(pb); /* reserved */

        get_be16(pb); /* reserved */

        get_be16(pb); /* index */



        /* for MPEG4: set codec type by looking for it */

        id = codec_get_id(mov_video_tags, format);

        if (id >= 0) {

            AVCodec *codec;

	    codec = avcodec_find_decoder(id);

            if (codec)

		st->codec.codec_type = codec->type;

        }

#ifdef DEBUG

        printf("size=%d 4CC= %c%c%c%c codec_type=%d\n",

               size,

               (format >> 0) & 0xff,

               (format >> 8) & 0xff,

               (format >> 16) & 0xff,

               (format >> 24) & 0xff,

               st->codec.codec_type);

#endif

	st->codec.codec_tag = format;

	if(st->codec.codec_type==CODEC_TYPE_VIDEO) {

	    MOV_atom_t a = { 0, 0, 0 };

            st->codec.codec_id = id;

            get_be16(pb); /* version */

            get_be16(pb); /* revision level */

            get_be32(pb); /* vendor */

            get_be32(pb); /* temporal quality */

            get_be32(pb); /* spacial quality */

            st->codec.width = get_be16(pb); /* width */

            st->codec.height = get_be16(pb); /* height */

#if 1

            if (st->codec.codec_id == CODEC_ID_MPEG4) {

                /* in some MPEG4 the width/height are not correct, so

                   we ignore this info */

                st->codec.width = 0;

                st->codec.height = 0;

            }

#endif

            get_be32(pb); /* horiz resolution */

            get_be32(pb); /* vert resolution */

            get_be32(pb); /* data size, always 0 */

            frames_per_sample = get_be16(pb); /* frames per samples */

#ifdef DEBUG

	    printf("frames/samples = %d\n", frames_per_sample);

#endif

	    get_buffer(pb, (uint8_t *)st->codec.codec_name, 32); /* codec name */



	    st->codec.bits_per_sample = get_be16(pb); /* depth */

            st->codec.color_table_id = get_be16(pb); /* colortable id */



            st->codec.frame_rate      = 25;

            st->codec.frame_rate_base = 1;



	    size -= (16+8*4+2+32+2*2);

#if 0

	    while (size >= 8) {

		MOV_atom_t a;

                int64_t start_pos;



		a.size = get_be32(pb);

		a.type = get_le32(pb);

		size -= 8;

#ifdef DEBUG

                printf("VIDEO: atom_type=%c%c%c%c atom.size=%Ld size_left=%d\n",

                       (a.type >> 0) & 0xff,

                       (a.type >> 8) & 0xff,

                       (a.type >> 16) & 0xff,

                       (a.type >> 24) & 0xff,

		       a.size, size);

#endif

                start_pos = url_ftell(pb);



		switch(a.type) {

                case MKTAG('e', 's', 'd', 's'):

                    {

                        int tag, len;

                        /* Well, broken but suffisant for some MP4 streams */

                        get_be32(pb); /* version + flags */

			len = mov_mp4_read_descr(pb, &tag);

                        if (tag == 0x03) {

                            /* MP4ESDescrTag */

                            get_be16(pb); /* ID */

                            get_byte(pb); /* priority */

			    len = mov_mp4_read_descr(pb, &tag);

                            if (tag != 0x04)

                                goto fail;

                            /* MP4DecConfigDescrTag */

                            get_byte(pb); /* objectTypeId */

                            get_be32(pb); /* streamType + buffer size */

			    get_be32(pb); /* max bit rate */

                            get_be32(pb); /* avg bit rate */

                            len = mp4_read_descr(pb, &tag);

                            if (tag != 0x05)

                                goto fail;

                            /* MP4DecSpecificDescrTag */

#ifdef DEBUG

                            printf("Specific MPEG4 header len=%d\n", len);

#endif

                            sc->header_data = av_mallocz(len);

                            if (sc->header_data) {

                                get_buffer(pb, sc->header_data, len);

				sc->header_len = len;

                            }

                        }

                        /* in any case, skip garbage */

                    }

                    break;

                default:

                    break;

                }

	    fail:

		printf("ATOMENEWSIZE %Ld   %d\n", atom.size, url_ftell(pb) - start_pos);

		if (atom.size > 8) {

		    url_fskip(pb, (atom.size - 8) -

			      ((url_ftell(pb) - start_pos)));

		    size -= atom.size - 8;

		}

	    }

            if (size > 0) {

                /* unknown extension */

                url_fskip(pb, size);

            }

#else

            a.size = size;

	    mov_read_default(c, pb, a);

#endif

	} else {

            get_be16(pb); /* version */

            get_be16(pb); /* revision level */

            get_be32(pb); /* vendor */



            st->codec.channels = get_be16(pb);		/* channel count */

	    st->codec.bits_per_sample = get_be16(pb);	/* sample size */



	    st->codec.codec_id = codec_get_id(mov_audio_tags, format);

            /* handle specific s8 codec */

            get_be16(pb); /* compression id = 0*/

            get_be16(pb); /* packet size = 0 */



            st->codec.sample_rate = ((get_be32(pb) >> 16));

	    //printf("CODECID %d  %d  %.4s\n", st->codec.codec_id, CODEC_ID_PCM_S16BE, (char*)&format);



	    switch (st->codec.codec_id) {

	    case CODEC_ID_PCM_S16BE:

		if (st->codec.bits_per_sample == 8)

		    st->codec.codec_id = CODEC_ID_PCM_S8;

                /* fall */

	    case CODEC_ID_PCM_U8:

		st->codec.bit_rate = st->codec.sample_rate * 8;

		break;

	    default:

                ;

	    }

	    get_be32(pb); /* samples per packet */

	    get_be32(pb); /* bytes per packet */

            get_be32(pb); /* bytes per frame */

            get_be32(pb); /* bytes per sample */



	    {

		MOV_atom_t a = { format, url_ftell(pb), size - (16 + 20 + 16 + 8) };

		mov_read_default(c, pb, a);

	    }

        }

    }



    return 0;

}
