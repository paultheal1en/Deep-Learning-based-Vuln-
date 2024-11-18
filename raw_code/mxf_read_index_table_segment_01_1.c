static int mxf_read_index_table_segment(MXFIndexTableSegment *segment, ByteIOContext *pb, int tag)

{

    switch(tag) {

    case 0x3F05: dprintf(NULL, "EditUnitByteCount %d\n", get_be32(pb)); break;

    case 0x3F06: dprintf(NULL, "IndexSID %d\n", get_be32(pb)); break;

    case 0x3F07: dprintf(NULL, "BodySID %d\n", get_be32(pb)); break;

    case 0x3F0B: dprintf(NULL, "IndexEditRate %d/%d\n", get_be32(pb), get_be32(pb)); break;

    case 0x3F0C: dprintf(NULL, "IndexStartPosition %lld\n", get_be64(pb)); break;

    case 0x3F0D: dprintf(NULL, "IndexDuration %lld\n", get_be64(pb)); break;

    }

    return 0;

}
