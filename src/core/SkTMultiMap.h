/*
 * Copyright 2013 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTMultiMap_DEFINED
#define SkTMultiMap_DEFINED

#include "src/core/SkTDynamicHash.h"

/** A set that contains pointers to instances of T. Instances can be looked up with key Key.
 * Multiple (possibly same) values can have the same key.
 */
template <typename T,
          typename Key,
          typename HashTraits=T>
class SkTMultiMap {
    struct ValueList {
        explicit ValueList(T* value) : fValue(value), fNext(nullptr), fCount(1) {}

        static const Key& GetKey(const ValueList& e) { return HashTraits::GetKey(*e.fValue); }
        static uint32_t Hash(const Key& key) { return HashTraits::Hash(key); }
        T* fValue;
        ValueList* fNext;

        // TODO(b/407062399): Debugging information holding count of elements in ValueList.
        // Only maintained in the head of the linked list to view in LLDB dumps.
        uint32_t fCount;
    };
public:
    SkTMultiMap() : fCount(0) {}

    ~SkTMultiMap() {
        this->reset();
    }

    void reset() {
        fHash.foreach([&](ValueList* vl) {
            ValueList* next;
            for (ValueList* it = vl; it; it = next) {
                HashTraits::OnFree(it->fValue);
                next = it->fNext;
                delete it;
            }
        });
        fHash.reset();
        fCount = 0;
    }

    void insert(const Key& key, T* value) {
        ValueList* list = fHash.find(key);
        if (list) {
            // The new ValueList entry is inserted as the second element in the
            // linked list, and it will contain the value of the first element.
            ValueList* newEntry = new ValueList(list->fValue);
            newEntry->fNext = list->fNext;
            // The existing first ValueList entry is updated to contain the
            // inserted value.
            list->fNext = newEntry;
            list->fValue = value;
            list->fCount++;
        } else {
            fHash.add(new ValueList(value));
        }

        ++fCount;
    }

    void remove(const Key& key, const T* value) {
        ValueList* root = fHash.find(key);
        ValueList* list = root;
        // Temporarily making this safe for remove entries not in the map because of
        // crbug.com/877915.
#if 0
        // Since we expect the caller to be fully aware of what is stored, just
        // assert that the caller removes an existing value.
        SkASSERT(list);
        ValueList* prev = nullptr;
        while (list->fValue != value) {
            prev = list;
            list = list->fNext;
        }
        this->internalRemove(root, prev, list, key);
#else
        ValueList* prev = nullptr;
        while (list && list->fValue != value) {
            prev = list;
            list = list->fNext;
        }
        // Crash in Debug since it'd be great to detect a repro of 877915.
        SkASSERT(list);
        if (list) {
            this->internalRemove(root, prev, list, key);
        }
#endif
    }

    T* find(const Key& key) const {
        ValueList* list = fHash.find(key);
        if (list) {
            return list->fValue;
        }
        return nullptr;
    }

    template<class FindPredicate>
    T* find(const Key& key, const FindPredicate f) {
        ValueList* list = fHash.find(key);
        while (list) {
            if (f(list->fValue)){
                return list->fValue;
            }
            list = list->fNext;
        }
        return nullptr;
    }

    template<class FindPredicate>
    T* findAndRemove(const Key& key, const FindPredicate f) {
        ValueList* root = fHash.find(key);

        ValueList* list = root;
        ValueList* prev = nullptr;
        while (list) {
            if (f(list->fValue)){
                T* value = list->fValue;
                this->internalRemove(root, prev, list, key);
                return value;
            }
            prev = list;
            list = list->fNext;
        }
        return nullptr;
    }

    int count() const { return fCount; }

#ifdef SK_DEBUG
    template <typename Fn>  // f(T) or f(const T&)
    void foreach(Fn&& fn) const {
        fHash.foreach([&](const ValueList& vl) {
            for (const ValueList* it = &vl; it; it = it->fNext) {
                fn(*it->fValue);
            }
        });
    }

    bool has(const T* value, const Key& key) const {
        for (ValueList* list = fHash.find(key); list; list = list->fNext) {
            if (list->fValue == value) {
                return true;
            }
        }
        return false;
    }

    // This is not particularly fast and only used for validation, so debug only.
    int countForKey(const Key& key) const {
        int count = 0;
        ValueList* list = fHash.find(key);
        while (list) {
            list = list->fNext;
            ++count;
        }
        return count;
    }
#endif

private:
    SkTDynamicHash<ValueList, Key> fHash;
    int fCount;

    void internalRemove(ValueList* root, ValueList* prev, ValueList* elem, const Key& key) {
        root->fCount--;
        if (elem->fNext) {
            ValueList* next = elem->fNext;
            elem->fValue = next->fValue;
            elem->fNext = next->fNext;
            delete next;
        } else if (prev) {
            prev->fNext = nullptr;
            delete elem;
        } else {
            fHash.remove(key);
            delete elem;
        }

        --fCount;
    }

};

#endif
