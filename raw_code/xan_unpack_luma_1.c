static int xan_unpack_luma(const uint8_t *src, const int src_size,

                           uint8_t *dst, const int dst_size)

{

   int tree_size, eof;

   const uint8_t *tree;

   int bits, mask;

   int tree_root, node;

   const uint8_t *dst_end = dst + dst_size;

   const uint8_t *src_end = src + src_size;



   tree_size = *src++;

   eof       = *src++;

   tree      = src - eof * 2 - 2;

   tree_root = eof + tree_size;

   src += tree_size * 2;



   node = tree_root;

   bits = *src++;

   mask = 0x80;

   for (;;) {

       int bit = !!(bits & mask);

       mask >>= 1;

       node = tree[node*2 + bit];

       if (node == eof)

           break;

       if (node < eof) {

           *dst++ = node;

           if (dst > dst_end)

               break;

           node = tree_root;

       }

       if (!mask) {

           bits = *src++;

           if (src > src_end)

               break;

           mask = 0x80;

       }

   }

   return dst != dst_end;

}
