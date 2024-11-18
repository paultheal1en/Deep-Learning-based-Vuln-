static int h263p_decode_umotion(MpegEncContext * s, int pred)

{

   int code = 0, sign;



   if (get_bits1(&s->gb)) /* Motion difference = 0 */

      return pred;



   code = 2 + get_bits1(&s->gb);



   while (get_bits1(&s->gb))

   {

      code <<= 1;

      code += get_bits1(&s->gb);

      if (code >= 32768) {

          avpriv_request_sample(s->avctx, "Huge DMV");

          return AVERROR_INVALIDDATA;

      }

   }

   sign = code & 1;

   code >>= 1;



   code = (sign) ? (pred - code) : (pred + code);

   ff_tlog(s->avctx,"H.263+ UMV Motion = %d\n", code);

   return code;



}
