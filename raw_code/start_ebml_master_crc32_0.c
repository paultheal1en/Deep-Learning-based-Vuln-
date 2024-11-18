static int start_ebml_master_crc32(AVIOContext *pb, AVIOContext **dyn_cp, ebml_master *master,

                                   unsigned int elementid, uint64_t expectedsize)

{

    int ret;



    if ((ret = avio_open_dyn_buf(dyn_cp)) < 0)

        return ret;



    if (pb->seekable)

        *master = start_ebml_master(pb, elementid, expectedsize);

    else

        *master = start_ebml_master(*dyn_cp, elementid, expectedsize);



    return 0;

}
