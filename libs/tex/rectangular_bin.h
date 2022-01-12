/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_RECTANGULARBIN_HEADER
#define TEX_RECTANGULARBIN_HEADER

#include <list>
#include <memory>

#include "rect.h"

/**
  * Implementation of the binpacking algorithm GUILLUTINE from
  * <a href="http://clb.demon.fi/files/RectangleBinPack.pdf">
  * A Thousand Ways to Pack the Bin -
  * A Practical Approach to Two-Dimensional Rectangle Bin Packing
  * </a>
  */
class RectangularBin {
    public:
        typedef std::shared_ptr<RectangularBin> Ptr;

    private:
        long width;
        long height;
        std::list<Rect<long> > rects;

    public:
        /**
          * Initializes the rectangular binpacking algorithm to fill a rectangle of the given size.
          */
        RectangularBin(long width, long height);

        static RectangularBin::Ptr create(long width, long height);

        /** Returns true and changes the position of the given rect if it fits into the bin. */
        bool insert(Rect<long> * rect);
};

inline RectangularBin::Ptr
RectangularBin::create(long width, long height)
{
    return Ptr(new RectangularBin(width, height));
}

#endif /* TEX_RECTANGULARBIN_HEADER */
