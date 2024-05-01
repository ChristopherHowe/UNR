export default function CurrentEvent({ heading, message }: { heading: string; message: string }) {
  return (
    <div className="w-full absolute bottom-0">
      <div className="m-4 p-3 bg-white shadow-sm ring-1 ring-gray-900/5 sm:rounded-xl">
        <div className="w-full flex flex-col items-center">
          <div className="xl:w-1/3">
            <h1 className="text-2xl">{heading}</h1>
            <p className="text-md">{message}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
