static void dmg_close(BlockDriverState *bs)

{

    BDRVDMGState *s = bs->opaque;



    g_free(s->types);

    g_free(s->offsets);

    g_free(s->lengths);

    g_free(s->sectors);

    g_free(s->sectorcounts);

    g_free(s->compressed_chunk);

    g_free(s->uncompressed_chunk);



    inflateEnd(&s->zstream);

}
