static void slirp_receive(void *opaque, const uint8_t *buf, size_t size)

{

#ifdef DEBUG_SLIRP

    printf("slirp input:\n");

    hex_dump(stdout, buf, size);

#endif

    slirp_input(buf, size);

}
