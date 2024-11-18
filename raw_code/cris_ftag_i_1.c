static inline void cris_ftag_i(unsigned int x)

{

	register unsigned int v asm("$r10") = x;

	asm ("ftagi\t[%0]\n" : : "r" (v) );

}
