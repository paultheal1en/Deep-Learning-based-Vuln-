static int mxf_write_footer(AVFormatContext *s)

{

    MXFContext *mxf = s->priv_data;

    AVIOContext *pb = s->pb;



    mxf->duration = mxf->last_indexed_edit_unit + mxf->edit_units_count;



    mxf_write_klv_fill(s);

    mxf->footer_partition_offset = avio_tell(pb);

    if (mxf->edit_unit_byte_count) { // no need to repeat index

        mxf_write_partition(s, 0, 0, footer_partition_key, 0);

    } else {

        mxf_write_partition(s, 0, 2, footer_partition_key, 0);



        mxf_write_klv_fill(s);

        mxf_write_index_table_segment(s);

    }



    mxf_write_klv_fill(s);

    mxf_write_random_index_pack(s);



    if (s->pb->seekable) {

        avio_seek(pb, 0, SEEK_SET);

        if (mxf->edit_unit_byte_count) {

            mxf_write_partition(s, 1, 2, header_closed_partition_key, 1);

            mxf_write_klv_fill(s);

            mxf_write_index_table_segment(s);

        } else {

            mxf_write_partition(s, 0, 0, header_closed_partition_key, 1);

        }

    }



    ff_audio_interleave_close(s);



    av_freep(&mxf->index_entries);

    av_freep(&mxf->body_partition_offset);

    av_freep(&mxf->timecode_track->priv_data);

    av_freep(&mxf->timecode_track);



    mxf_free(s);



    return 0;

}
