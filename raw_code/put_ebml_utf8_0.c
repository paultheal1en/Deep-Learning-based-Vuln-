static void put_ebml_utf8(ByteIOContext *pb, unsigned int elementid, char *str)

{

    put_ebml_binary(pb, elementid, str, strlen(str));

}
