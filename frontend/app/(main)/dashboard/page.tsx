"use client"

import { useEffect, useState } from "react"
import Papa from "papaparse"
import Image from "next/image"
import Link from "next/link"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

interface Obat {
  "Nama Obat": string
  "Link Obat": string
  "Link Gambar": string
  "Check": string
  "Aturan Pakai": string
  "Dosis": string
  "Efek Samping": string
  "Golongan Produk": string
  "Indikasi Umum": string
  "Kemasan": string
  "Komposisi": string
  "Kontra Indikasi": string
  "Manufaktur": string
  "No. Registrasi": string
  "Perhatian": string
  "Deskripsi": string
}


export default function DashboardPage() {
  const [data, setData] = useState<Obat[]>([])
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(1)
  const itemsPerPage = 6 * 4 // 6 baris × 4 kolom

  useEffect(() => {
    Papa.parse("/data/obat.csv", {
      download: true,
      header: true,
      complete: (results) => {
        const filtered = results.data.filter((d: any) => d["Check"] === "1")
        setData(filtered as Obat[])
      }
    })
  }, [])

  const filteredData = data.filter((item) =>
    item["Nama Obat"].toLowerCase().includes(search.toLowerCase())
  )

  const totalPages = Math.ceil(filteredData.length / itemsPerPage)
  const paginated = filteredData.slice(
    (page - 1) * itemsPerPage,
    page * itemsPerPage
  )


  function getVisiblePages(current: number, total: number): (number | "...")[] {
    const delta = 2
    const range: (number | "...")[] = []
    const left = Math.max(2, current - delta)
    const right = Math.min(total - 1, current + delta)

    range.push(1)
    if (left > 2) range.push("...")

    for (let i = left; i <= right; i++) {
      range.push(i)
    }

    if (right < total - 1) range.push("...")
    if (total > 1) range.push(total)

    return range
  }

  const [selectedObat, setSelectedObat] = useState<Obat | null>(null)
  const Field = ({ label, value }: { label: string; value: string }) => (
    <div>
      <span className="font-semibold">{label}:</span>{" "}
      <span className="text-gray-800">{value || "-"}</span>
    </div>
  )

  const InfoRow = ({
    label,
    value,
  }: {
    label: string
    value: React.ReactNode
  }) => (
    <tr className="border-t border-gray-200">
      <td className="font-medium text-gray-600 px-3 py-2 w-40 align-top">{label}</td>
      <td className="text-gray-800 px-3 py-2 whitespace-pre-line">{value || "-"}</td>
    </tr>
  )

  return (
    <div className="p-4">
      <div className="mb-4 flex justify-between items-center">
        <Input
          placeholder="Cari nama obat..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value)
            setPage(1)
          }}
          className="max-w-sm"
        />
        <span className="text-sm text-muted-foreground">
          Halaman {page} dari {totalPages}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-4">
        {paginated.map((obat, i) => (
          <Card key={i} className="rounded-lg h-[250px] flex flex-col justify-between">
            <CardContent className="p-3">
              <Image
                src={obat["Link Gambar"] || "/fallback.jpg"}
                alt={obat["Nama Obat"]}
                width={400}
                height={240}
                className="w-full h-24 object-contain rounded-md mb-2"
                unoptimized
              />
              <h2 className="text-sm font-semibold mb-1 line-clamp-2 h-[36px] leading-snug">
                {obat["Nama Obat"]}
              </h2>
              <p className="text-xs text-muted-foreground mb-3 truncate h-[18px] leading-tight">
                {(obat["No. Registrasi"] || "No Info").replace(/^BPOM:\s*/i, "")}
              </p>
              <div className="mt-auto">
                {/* <Link href={obat["Link Obat"]} target="_blank">
                  <Button size="sm" className="w-full">Lihat Detail</Button>
                </Link> */}
                <Button
                  variant="secondary"
                  size="sm"
                  className="w-full"
                  onClick={() => setSelectedObat(obat)}
                >
                  Lihat Detail
                </Button>

              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center mt-6 flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={page === 1}
            onClick={() => setPage((p) => p - 1)}
          >
            Prev
          </Button>
          {getVisiblePages(page, totalPages).map((p, idx) => (
            <Button
              key={idx}
              size="sm"
              variant={p === page ? "secondary" : "outline"}

              disabled={p === "..."}
              onClick={() => typeof p === "number" && setPage(p)}
            >
              {p}
            </Button>
          ))}

          <Button
            size="sm"
            variant="outline"
            disabled={page === totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      )}
      {selectedObat && (
        <div
          onClick={(e) => {
            if (e.target === e.currentTarget) setSelectedObat(null)
          }}
          className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/10"
        >
          <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full p-6 relative overflow-y-auto max-h-[90vh] border border-gray-200">
            {/* <button
              className="absolute top-3 right-3 text-gray-400 hover:text-black"
              onClick={() => setSelectedObat(null)}
            >
              ✕
            </button> */}
            <Button
              variant="destructive"
              size="sm"
              className="absolute top-3 right-3"
              onClick={() => setSelectedObat(null)}
            >
              X
            </Button>

            {/* Judul */}
            <h2 className="text-xl font-bold text-center mb-4 text-primary">
              {selectedObat["Nama Obat"]}
            </h2>

            {/* Gambar */}
            <div className="w-full mb-6 flex justify-center">
              <Image
                src={selectedObat["Link Gambar"] || "/fallback.jpg"}
                alt={selectedObat["Nama Obat"]}
                width={300}
                height={300}
                className="object-contain w-auto max-h-[250px] rounded-lg border"
                unoptimized
              />
            </div>

            {/* Tabel Informasi */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm border border-gray-300 rounded-md overflow-hidden">
                <tbody className="[&>tr:nth-child(even)]:bg-gray-50">
                  <InfoRow label="No. Registrasi" value={selectedObat["No. Registrasi"]} />
                  <InfoRow label="Manufaktur" value={selectedObat["Manufaktur"]} />
                  <InfoRow label="Golongan Produk" value={selectedObat["Golongan Produk"]} />
                  <InfoRow label="Indikasi Umum" value={selectedObat["Indikasi Umum"]} />
                  <InfoRow label="Kemasan" value={selectedObat["Kemasan"]} />
                  <InfoRow label="Komposisi" value={selectedObat["Komposisi"]} />
                  <InfoRow label="Dosis" value={selectedObat["Dosis"]} />
                  <InfoRow label="Aturan Pakai" value={selectedObat["Aturan Pakai"]} />
                  <InfoRow label="Efek Samping" value={selectedObat["Efek Samping"]} />
                  <InfoRow label="Kontra Indikasi" value={selectedObat["Kontra Indikasi"]} />
                  <InfoRow label="Perhatian" value={selectedObat["Perhatian"]} />
                  <InfoRow label="Deskripsi" value={selectedObat["Deskripsi"]} />
                  <InfoRow
                    label="Link"
                    value={
                      <a
                        href={selectedObat["Link Obat"]}
                        target="_blank"
                        className="text-blue-600 underline break-words"
                      >
                        Buka Halaman
                      </a>
                    }
                  />
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
