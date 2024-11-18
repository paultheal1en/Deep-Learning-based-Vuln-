void check_aligned_anonymous_unfixed_colliding_mmaps(void)

{

	char *p1;

	char *p2;

	char *p3;

	uintptr_t p;

	int i;



	fprintf (stderr, "%s", __func__);

	for (i = 0; i < 0x2fff; i++)

	{

		int nlen;

		p1 = mmap(NULL, pagesize, PROT_READ, 

			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

		fail_unless (p1 != MAP_FAILED);

		p = (uintptr_t) p1;

		fail_unless ((p & pagemask) == 0);

		memcpy (dummybuf, p1, pagesize);



		p2 = mmap(NULL, pagesize, PROT_READ, 

			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

		fail_unless (p2 != MAP_FAILED);

		p = (uintptr_t) p2;

		fail_unless ((p & pagemask) == 0);

		memcpy (dummybuf, p2, pagesize);





		munmap (p1, pagesize);

		nlen = pagesize * 8;

		p3 = mmap(NULL, nlen, PROT_READ, 

			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);



		/* Check if the mmaped areas collide.  */

		if (p3 < p2 

		    && (p3 + nlen) > p2)

			fail_unless (0);



		memcpy (dummybuf, p3, pagesize);



		/* Make sure we get pages aligned with the pagesize. The

		   target expects this.  */

		fail_unless (p3 != MAP_FAILED);

		p = (uintptr_t) p3;

		fail_unless ((p & pagemask) == 0);

		munmap (p2, pagesize);

		munmap (p3, nlen);

	}

	fprintf (stderr, " passed\n");

}
