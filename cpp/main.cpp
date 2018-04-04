#include <iostream>

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

struct WSNode {
    int next;
    int mask_ofs;
    int img_ofs;
};

struct WSQueue {
    WSQueue() { first = last = 0; }

    int first, last;
};

static int allocWSNodes(std::vector<WSNode> &storage) {
    int sz = (int) storage.size();
    int newsz = MAX(128, sz * 3 / 2);

    storage.resize(newsz);
    if (sz == 0) {
        storage[0].next = 0;
        sz = 1;
    }
    for (int i = sz; i < newsz - 1; i++)
        storage[i].next = i + 1;
    storage[newsz - 1].next = 0;
    return sz;
}

void watershed(InputArray _src, InputOutputArray _markers) {
    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    const int WSHED = -1; // Pixel belongs to watershed

    // possible bit values = 2^8
    const int NQ = 256;

    Mat src = _src.getMat(), dst = _markers.getMat();
    Size size = src.size();

    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a, b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a, b) ((a) - subs_tab[(a)-(b)+NQ])

    // Create a new node with offsets mofs and iofs in queue idx
#define ws_push(idx, mofs, iofs)                    \
    {                                               \
        if( !free_node )                            \
            free_node = allocWSNodes( storage );    \
        node = free_node;                           \
        free_node = storage[free_node].next;        \
        storage[node].next = 0;                     \
        storage[node].mask_ofs = mofs;              \
        storage[node].img_ofs = iofs;               \
        if( q[idx].last )                           \
            storage[q[idx].last].next=node;         \
        else                                        \
            q[idx].first = node;                    \
        q[idx].last = node;                         \
    }

    // Get next node from queue idx
#define ws_pop(idx, mofs, iofs)                     \
    {                                               \
        node = q[idx].first;                        \
        q[idx].first = storage[node].next;          \
        if( !storage[node].next )                   \
            q[idx].last = 0;                        \
        storage[node].next = free_node;             \
        free_node = node;                           \
        (mofs) = storage[node].mask_ofs;              \
        (iofs) = storage[node].img_ofs;               \
    }

    // Get highest absolute channel difference in diff
#define c_diff(ptr1, ptr2, diff)                    \
    {                                               \
        db = std::abs((ptr1)[0] - (ptr2)[0]);       \
        dg = std::abs((ptr1)[1] - (ptr2)[1]);       \
        dr = std::abs((ptr1)[2] - (ptr2)[2]);       \
        (diff) = ws_max(db,dg);                       \
        diff = ws_max(diff,dr);                     \
        assert( 0 <= diff && diff <= 255 );         \
    }


    // Current pixel in input image
    const uchar *img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step / sizeof(img[0]));

    // Current pixel in mask image
    int *mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for (i = 0; i < 256; i++)
        subs_tab[i] = 0;
    for (i = 256; i <= 512; i++)
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for (j = 0; j < size.width; j++)
        mask[j] = mask[j + mstep * (size.height - 1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for (i = 1; i < size.height - 1; i++) {
        img += istep;
        mask += mstep;
        mask[0] = mask[size.width - 1] = WSHED; // boundary pixels

        for (j = 1; j < size.width - 1; j++) {
            int *m = mask + j;
            if (m[0] < 0) m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0)) {
                // Find smallest difference to adjacent markers
                const uchar *ptr = img + j * 3;
                int idx = 256, t;
                if (m[-1] > 0) c_diff(ptr, ptr - 3, idx);
                if (m[1] > 0) {
                    c_diff(ptr, ptr + 3, t);
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0) {
                    c_diff(ptr, ptr - istep, t);
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0) {
                    c_diff(ptr, ptr + istep, t);
                    idx = ws_min(idx, t);
                }

                // Add to according queue
                assert(0 <= idx && idx <= 255);
                ws_push(idx, i * mstep + j, i * istep + j * 3);
                m[0] = IN_QUEUE;
            }
        }
    }

    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
        if (q[i].first)
            break;

    // if there is no markers, exit immediately
    if (i == NQ)
        return;

    active_queue = i;
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for (;;) {
        int mofs, iofs;
        int lab = 0, t;
        int *m;
        const uchar *ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if (q[active_queue].first == 0) {
            for (i = active_queue + 1; i < NQ; i++)
                if (q[i].first)
                    break;
            if (i == NQ)
                break;
            active_queue = i;
        }

        // Get next node
        ws_pop(active_queue, mofs, iofs);

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;

        // Check surrounding pixels for labels
        // to determine label for current pixel
        t = m[-1]; // Left
        if (t > 0) lab = t;
        t = m[1]; // Right
        if (t > 0) {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[-mstep]; // Top
        if (t > 0) {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[mstep]; // Bottom
        if (t > 0) {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }

        // Set label to current pixel in marker image
        assert(lab != 0);
        m[0] = lab;

        if (lab == WSHED)
            continue;

        // Add adjacent, unlabeled pixels to corresponding queue
        if (m[-1] == 0) {
            c_diff(ptr, ptr - 3, t);
            ws_push(t, mofs - 1, iofs - 3);
            active_queue = ws_min(active_queue, t);
            m[-1] = IN_QUEUE;
        }
        if (m[1] == 0) {
            c_diff(ptr, ptr + 3, t);
            ws_push(t, mofs + 1, iofs + 3);
            active_queue = ws_min(active_queue, t);
            m[1] = IN_QUEUE;
        }
        if (m[-mstep] == 0) {
            c_diff(ptr, ptr - istep, t);
            ws_push(t, mofs - mstep, iofs - istep);
            active_queue = ws_min(active_queue, t);
            m[-mstep] = IN_QUEUE;
        }
        if (m[mstep] == 0) {
            c_diff(ptr, ptr + istep, t);
            ws_push(t, mofs + mstep, iofs + istep);
            active_queue = ws_min(active_queue, t);
            m[mstep] = IN_QUEUE;
        }
    }
}


int main(int argc, char **argv) {
    Mat image = imread("fruits.png", IMREAD_COLOR);
    Mat markers = imread("markers.png", IMREAD_GRAYSCALE);
    markers.convertTo(markers, CV_32SC1);

    watershed(image, markers);
    imwrite("output.png", markers);

    return 0;
}
