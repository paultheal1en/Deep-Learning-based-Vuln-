static void mxf_write_generic_desc(ByteIOContext *pb, const MXFDescriptorWriteTableEntry *desc_tbl, AVStream *st)

{

    const MXFCodecUL *codec_ul;



    put_buffer(pb, desc_tbl->key, 16);

    klv_encode_ber_length(pb, 108);



    mxf_write_local_tag(pb, 16, 0x3C0A);

    mxf_write_uuid(pb, SubDescriptor, st->index);



    mxf_write_local_tag(pb, 4, 0x3006);

    put_be32(pb, st->index);



    mxf_write_local_tag(pb, 8, 0x3001);

    put_be32(pb, st->time_base.den);

    put_be32(pb, st->time_base.num);



    codec_ul = mxf_get_essence_container_ul(st->codec->codec_id);

    mxf_write_local_tag(pb, 16, 0x3004);

    put_buffer(pb, codec_ul->uid, 16);

}
