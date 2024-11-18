static void init_dev(tc58128_dev * dev, const char *filename)

{

    int ret, blocks;



    dev->state = WAIT;

    dev->flash_contents = g_malloc0(FLASH_SIZE);

    memset(dev->flash_contents, 0xff, FLASH_SIZE);

    if (!dev->flash_contents) {

	fprintf(stderr, "could not alloc memory for flash\n");

	exit(1);

    }

    if (filename) {

	/* Load flash image skipping the first block */

	ret = load_image(filename, dev->flash_contents + 528 * 32);

	if (ret < 0) {

	    fprintf(stderr, "ret=%d\n", ret);

	    fprintf(stderr, "qemu: could not load flash image %s\n",

		    filename);

	    exit(1);

	} else {

	    /* Build first block with number of blocks */

	    blocks = (ret + 528 * 32 - 1) / (528 * 32);

	    dev->flash_contents[0] = blocks & 0xff;

	    dev->flash_contents[1] = (blocks >> 8) & 0xff;

	    dev->flash_contents[2] = (blocks >> 16) & 0xff;

	    dev->flash_contents[3] = (blocks >> 24) & 0xff;

	    fprintf(stderr, "loaded %d bytes for %s into flash\n", ret,

		    filename);

	}

    }

}
