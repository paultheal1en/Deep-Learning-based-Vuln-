static void test_cdrom_dma(void)

{

    static const size_t len = ATAPI_BLOCK_SIZE;

    char *pattern = g_malloc(ATAPI_BLOCK_SIZE * 16);

    char *rx = g_malloc0(len);

    uintptr_t guest_buf;

    PrdtEntry prdt[1];

    FILE *fh;



    ide_test_start("-drive if=none,file=%s,media=cdrom,format=raw,id=sr0,index=0 "

                   "-device ide-cd,drive=sr0,bus=ide.0", tmp_path);

    qtest_irq_intercept_in(global_qtest, "ioapic");



    guest_buf = guest_alloc(guest_malloc, len);

    prdt[0].addr = cpu_to_le32(guest_buf);

    prdt[0].size = cpu_to_le32(len | PRDT_EOT);



    generate_pattern(pattern, ATAPI_BLOCK_SIZE * 16, ATAPI_BLOCK_SIZE);

    fh = fopen(tmp_path, "w+");

    fwrite(pattern, ATAPI_BLOCK_SIZE, 16, fh);

    fclose(fh);



    send_dma_request(CMD_PACKET, 0, 1, prdt, 1, send_scsi_cdb_read10);



    /* Read back data from guest memory into local qtest memory */

    memread(guest_buf, rx, len);

    g_assert_cmpint(memcmp(pattern, rx, len), ==, 0);



    g_free(pattern);

    g_free(rx);

    test_bmdma_teardown();

}
