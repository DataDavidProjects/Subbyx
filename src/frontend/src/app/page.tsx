import Link from "next/link";

export default function Home() {
  return (
    <main className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Subbyx</h1>
          <p className="text-gray-600 mt-2">Checkout Fraud Prediction System</p>
        </div>

        <div className="grid grid-cols-1">
          <Link
            href="/realtime"
            className="block p-6 border-2 border-gray-200 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition"
          >
            <div className="text-4xl mb-4">⚡</div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Realtime Prediction</h2>
            <p className="text-gray-600 text-sm">
              Select a checkout from future data and get instant fraud prediction with features
            </p>
          </Link>
        </div>
      </div>
    </main>
  );
}
