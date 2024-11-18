m_free(struct mbuf *m)

{



  DEBUG_CALL("m_free");

  DEBUG_ARG("m = %lx", (long )m);



  if(m) {

	/* Remove from m_usedlist */

	if (m->m_flags & M_USEDLIST)

	   remque(m);



	/* If it's M_EXT, free() it */

	if (m->m_flags & M_EXT)

	   free(m->m_ext);



	/*

	 * Either free() it or put it on the free list

	 */

	if (m->m_flags & M_DOFREE) {

		free(m);

		m->slirp->mbuf_alloced--;

	} else if ((m->m_flags & M_FREELIST) == 0) {

		insque(m,&m->slirp->m_freelist);

		m->m_flags = M_FREELIST; /* Clobber other flags */

	}

  } /* if(m) */

}
