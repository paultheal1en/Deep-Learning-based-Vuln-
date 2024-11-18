void check_file_fixed_eof_mmaps(void)

{

	char *addr;

	char *cp;

	unsigned int *p1;

	uintptr_t p;

	int i;



	/* Find a suitable address to start with.  */

	addr = mmap(NULL, pagesize * 44, PROT_READ, 

		    MAP_PRIVATE | MAP_ANONYMOUS,

		    -1, 0);



	fprintf (stderr, "%s addr=%p", __func__, (void *)addr);

	fail_unless (addr != MAP_FAILED);



	for (i = 0; i < 0x10; i++)

	{

		/* Create submaps within our unfixed map.  */

		p1 = mmap(addr, pagesize, PROT_READ, 

			  MAP_PRIVATE | MAP_FIXED, 

			  test_fd, 

			  (test_fsize - sizeof *p1) & ~pagemask);



		fail_unless (p1 != MAP_FAILED);



		/* Make sure we get pages aligned with the pagesize. The

		   target expects this.  */

		p = (uintptr_t) p1;

		fail_unless ((p & pagemask) == 0);



		/* Verify that the file maps was made correctly.  */

		fail_unless (p1[(test_fsize & pagemask) / sizeof *p1 - 1]

			     == ((test_fsize - sizeof *p1) / sizeof *p1));



		/* Verify that the end of page is accessable and zeroed.  */

		cp = (void *)p1;

		fail_unless (cp[pagesize - 4] == 0);

		munmap (p1, pagesize);

		addr += pagesize;

	}

	fprintf (stderr, " passed\n");

}
