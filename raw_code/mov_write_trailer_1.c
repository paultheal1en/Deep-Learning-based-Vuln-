static int mov_write_trailer(AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    AVIOContext *pb = s->pb;

    int res = 0;

    int i;



    int64_t moov_pos = avio_tell(pb);



    if (!(mov->flags & FF_MOV_FLAG_FRAGMENT)) {

        /* Write size of mdat tag */

        if (mov->mdat_size + 8 <= UINT32_MAX) {

            avio_seek(pb, mov->mdat_pos, SEEK_SET);

            avio_wb32(pb, mov->mdat_size + 8);

        } else {

            /* overwrite 'wide' placeholder atom */

            avio_seek(pb, mov->mdat_pos - 8, SEEK_SET);

            /* special value: real atom size will be 64 bit value after

             * tag field */

            avio_wb32(pb, 1);

            ffio_wfourcc(pb, "mdat");

            avio_wb64(pb, mov->mdat_size + 16);

        }

        avio_seek(pb, mov->reserved_moov_size ? mov->reserved_moov_pos : moov_pos, SEEK_SET);



        mov_write_moov_tag(pb, mov, s);

        if(mov->reserved_moov_size){

            int64_t size=  mov->reserved_moov_size - (avio_tell(pb) - mov->reserved_moov_pos);

            if(size < 8){

                av_log(s, AV_LOG_ERROR, "reserved_moov_size is too small, needed %"PRId64" additional\n", 8-size);

                return -1;

            }

            avio_wb32(pb, size);

            ffio_wfourcc(pb, "free");

            for(i=0; i<size; i++)

                avio_w8(pb, 0);

            avio_seek(pb, moov_pos, SEEK_SET);

        }

    } else {

        mov_flush_fragment(s);

        mov_write_mfra_tag(pb, mov);

    }



    if (mov->chapter_track)

        av_freep(&mov->tracks[mov->chapter_track].enc);



    for (i=0; i<mov->nb_streams; i++) {

        if (mov->tracks[i].tag == MKTAG('r','t','p',' '))

            ff_mov_close_hinting(&mov->tracks[i]);

        else if (mov->tracks[i].tag == MKTAG('t','m','c','d'))

            av_freep(&mov->tracks[i].enc);

        if (mov->flags & FF_MOV_FLAG_FRAGMENT &&

            mov->tracks[i].vc1_info.struct_offset && s->pb->seekable) {

            int64_t off = avio_tell(pb);

            uint8_t buf[7];

            if (mov_write_dvc1_structs(&mov->tracks[i], buf) >= 0) {

                avio_seek(pb, mov->tracks[i].vc1_info.struct_offset, SEEK_SET);

                avio_write(pb, buf, 7);

                avio_seek(pb, off, SEEK_SET);

            }

        }

        av_freep(&mov->tracks[i].cluster);

        av_freep(&mov->tracks[i].frag_info);



        if (mov->tracks[i].vos_len)

            av_free(mov->tracks[i].vos_data);



    }



    avio_flush(pb);



    av_freep(&mov->tracks);



    return res;

}
