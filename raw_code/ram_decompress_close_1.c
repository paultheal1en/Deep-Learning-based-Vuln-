static void ram_decompress_close(RamDecompressState *s)

{

    inflateEnd(&s->zstream);

}
