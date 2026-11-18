import pytest

from ncm.text import BPE

SAMPLE_CODE = [
    """package com.example.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class GreetingController {
    @GetMapping("/greeting")
    public String greeting() {
        return "hello";
    }
}
""",
    """package main

import (
\t"fmt"
\t"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
\tfmt.Fprintf(w, "ok")
}

func main() {
\thttp.HandleFunc("/", handler)
\thttp.ListenAndServe(":8080", nil)
}
""",
    """import express from "express";

const app = express();
app.get("/health", (req, res) => res.json({ status: "ok" }));
app.listen(3000);
""",
]


@pytest.fixture(scope="session")
def bpe() -> BPE:
    return BPE.train(SAMPLE_CODE * 4, vocab_size=400)
