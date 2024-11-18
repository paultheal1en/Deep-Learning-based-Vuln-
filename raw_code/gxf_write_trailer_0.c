static int gxf_write_trailer(AVFormatContext *s)

{

    GXFContext *gxf = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t end;

    int i;



    ff_audio_interleave_close(s);



    gxf_write_eos_packet(pb);

    end = avio_tell(pb);

    avio_seek(pb, 0, SEEK_SET);

    /* overwrite map, flt and umf packets with new values */

    gxf_write_map_packet(s, 1);

    gxf_write_flt_packet(s);

    gxf_write_umf_packet(s);

    avio_flush(pb);

    /* update duration in all map packets */

    for (i = 1; i < gxf->map_offsets_nb; i++) {

        avio_seek(pb, gxf->map_offsets[i], SEEK_SET);

        gxf_write_map_packet(s, 1);

        avio_flush(pb);

    }



    avio_seek(pb, end, SEEK_SET);



    av_freep(&gxf->flt_entries);

    av_freep(&gxf->map_offsets);



    return 0;

}
