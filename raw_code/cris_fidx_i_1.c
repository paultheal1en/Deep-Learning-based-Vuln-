static inline void cris_fidx_i(unsigned int x)

{

	register unsigned int v asm("$r10") = x;

	asm ("fidxi\t[%0]\n" : : "r" (v) );

}
