static inline void cris_ftag_d(unsigned int x)

{

	register unsigned int v asm("$r10") = x;

	asm ("ftagd\t[%0]\n" : : "r" (v) );

}
