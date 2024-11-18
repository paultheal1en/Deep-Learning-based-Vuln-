static void qxl_init_ramsize(PCIQXLDevice *qxl)

{

    /* vga mode framebuffer / primary surface (bar 0, first part) */

    if (qxl->vgamem_size_mb < 8) {

        qxl->vgamem_size_mb = 8;








    qxl->vgamem_size = qxl->vgamem_size_mb * 1024 * 1024;



    /* vga ram (bar 0, total) */

    if (qxl->ram_size_mb != -1) {

        qxl->vga.vram_size = qxl->ram_size_mb * 1024 * 1024;


    if (qxl->vga.vram_size < qxl->vgamem_size * 2) {

        qxl->vga.vram_size = qxl->vgamem_size * 2;




    /* vram32 (surfaces, 32bit, bar 1) */

    if (qxl->vram32_size_mb != -1) {

        qxl->vram32_size = qxl->vram32_size_mb * 1024 * 1024;


    if (qxl->vram32_size < 4096) {

        qxl->vram32_size = 4096;




    /* vram (surfaces, 64bit, bar 4+5) */

    if (qxl->vram_size_mb != -1) {

        qxl->vram_size = qxl->vram_size_mb * 1024 * 1024;


    if (qxl->vram_size < qxl->vram32_size) {

        qxl->vram_size = qxl->vram32_size;




    if (qxl->revision == 1) {

        qxl->vram32_size = 4096;

        qxl->vram_size = 4096;


    qxl->vgamem_size = msb_mask(qxl->vgamem_size * 2 - 1);

    qxl->vga.vram_size = msb_mask(qxl->vga.vram_size * 2 - 1);

    qxl->vram32_size = msb_mask(qxl->vram32_size * 2 - 1);

    qxl->vram_size = msb_mask(qxl->vram_size * 2 - 1);
