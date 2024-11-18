static int vvfat_open(BlockDriverState *bs, const char* dirname, int flags)

{

    BDRVVVFATState *s = bs->opaque;

    int floppy = 0;

    int i;



#ifdef DEBUG

    vvv = s;

#endif



DLOG(if (stderr == NULL) {

    stderr = fopen("vvfat.log", "a");

    setbuf(stderr, NULL);

})



    s->bs = bs;



    s->fat_type=16;

    /* LATER TODO: if FAT32, adjust */

    s->sectors_per_cluster=0x10;

    /* 504MB disk*/

    bs->cyls=1024; bs->heads=16; bs->secs=63;



    s->current_cluster=0xffffffff;



    s->first_sectors_number=0x40;

    /* read only is the default for safety */

    bs->read_only = 1;

    s->qcow = s->write_target = NULL;

    s->qcow_filename = NULL;

    s->fat2 = NULL;

    s->downcase_short_names = 1;



    if (!strstart(dirname, "fat:", NULL))

	return -1;



    if (strstr(dirname, ":floppy:")) {

	floppy = 1;

	s->fat_type = 12;

	s->first_sectors_number = 1;

	s->sectors_per_cluster=2;

	bs->cyls = 80; bs->heads = 2; bs->secs = 36;

    }



    if (strstr(dirname, ":32:")) {

	fprintf(stderr, "Big fat greek warning: FAT32 has not been tested. You are welcome to do so!\n");

	s->fat_type = 32;

    } else if (strstr(dirname, ":16:")) {

	s->fat_type = 16;

    } else if (strstr(dirname, ":12:")) {

	s->fat_type = 12;

	bs->secs = 18;

    }



    s->sector_count=bs->cyls*bs->heads*bs->secs-(s->first_sectors_number-1);



    if (strstr(dirname, ":rw:")) {

	if (enable_write_target(s))

	    return -1;

	bs->read_only = 0;

    }



    i = strrchr(dirname, ':') - dirname;

    assert(i >= 3);

    if (dirname[i-2] == ':' && qemu_isalpha(dirname[i-1]))

	/* workaround for DOS drive names */

	dirname += i-1;

    else

	dirname += i+1;



    bs->total_sectors=bs->cyls*bs->heads*bs->secs;



    if(init_directories(s, dirname))

	return -1;



    s->sector_count = s->faked_sectors + s->sectors_per_cluster*s->cluster_count;



    if(s->first_sectors_number==0x40)

	init_mbr(s);



    /* for some reason or other, MS-DOS does not like to know about CHS... */

    if (floppy)

	bs->heads = bs->cyls = bs->secs = 0;



    //    assert(is_consistent(s));

    qemu_co_mutex_init(&s->lock);

    return 0;

}
